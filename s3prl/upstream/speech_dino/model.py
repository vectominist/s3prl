import torch
import torch.nn.functional as F


def extract_feat(
    self,
    source,
    padding_mask=None,
    get_teacher=False,
    get_vq=False,
    get_pred=False,
    layer=None,
):
    # assume that the model is in eval and no_grad

    if get_vq:
        get_teacher = True

    results = {}

    features = source

    features = self.feature_extractor(features)
    features = features.transpose(1, 2)
    features = self.layer_norm(features)

    orig_padding_mask = padding_mask

    if padding_mask is not None and padding_mask.any():
        input_lengths = (1 - padding_mask.long()).sum(-1)
        # apply conv formula to get real output_lengths
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        padding_mask = torch.zeros(
            features.shape[:2], dtype=features.dtype, device=features.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
    else:
        padding_mask = None

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    pre_encoder_features = None
    if self.cfg.ema_transformer_only:
        if self.pre_encoder_copied:
            self.cnn_copy.model.eval().to(source.device)
            self.ln_copy.model.eval().to(source.device)
            self.proj_copy.model.eval().to(source.device)
            with torch.no_grad():
                pre_encoder_features = self.cnn_copy.model(source)
                pre_encoder_features = pre_encoder_features.transpose(1, 2)
                pre_encoder_features = self.ln_copy.model(pre_encoder_features)
                pre_encoder_features = self.proj_copy.model(pre_encoder_features)
        else:
            pre_encoder_features = features.clone()

    features = self.dropout_input(features)

    x = features

    x, layer_results = self.encoder(
        x,
        padding_mask=padding_mask,
        layer=layer,
        min_layer=0,
    )

    results["student_feats"] = [features] + [
        _x.transpose(0, 1) for _x, _, _ in layer_results
    ]

    # for r in results["student_feats"]:
    #     print(r.shape)

    # assert all((f.shape[0] == 1 and f.shape[2] == 768) for f in results["student_feats"])

    if get_teacher:
        self.ema.model.eval()
        self.ema.model.to(source.device)

        if self.cfg.ema_transformer_only:
            y, layer_results = self.ema.model.extract_features(
                pre_encoder_features,
                padding_mask=padding_mask,
                min_layer=self.cfg.encoder_layers - self.top_k_layers,
            )
            y = {
                "x": y,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }
        else:
            y = self.ema.model.extract_features(
                source=source,
                padding_mask=orig_padding_mask,
                mask=False,
                min_layer=self.cfg.encoder_layers - self.top_k_layers,
            )

        # TBC -> BCT -> BTC
        target_layer_results = [l[2] for l in y["layer_results"]]
        target_layer_results = [
            tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
        ]
        target_layer_results = [
            F.instance_norm(tl.float()) for tl in target_layer_results
        ]
        target_layer_results = [
            tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
        ]

        if self.avg_target:
            target_layer_results = [
                sum(target_layer_results) / len(target_layer_results)
            ]

        results["teacher_feats"] = target_layer_results

    if get_vq:
        # Quantize targets & compute loss
        if self.codebooks[0].device != x.device:
            self.move_codebook_to_gpu()

        results["vq_onehot"] = []
        results["vq_labels"] = []
        for i, target in enumerate(target_layer_results):
            # Quantize target
            B, T, D = target.shape
            target = target.view(B * T, D)
            if self.advance_ema:
                codebook = self.codebooks[i].float() / self.codebook_cnts[i].unsqueeze(
                    1
                )
            else:
                codebook = self.codebooks[i].float()  # VxD
            V = codebook.size(0)
            # neg_l2_dist = - (target.unsqueeze(1).expand(-1, V, -1) - codebook.unsqueeze(0).expand(B*T, -1, -1)).pow(2).sum(-1)
            neg_l2_dist = -(
                torch.sum(target**2, dim=1, keepdim=True)
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(target, codebook.t())
            )
            if self.soft_target:
                onehot_target = torch.softmax(neg_l2_dist / self.teacher_temp, dim=-1)
            else:
                onehot_target = torch.zeros_like(neg_l2_dist)
                onehot_target[range(len(neg_l2_dist)), neg_l2_dist.argmax(-1)] = 1.0

            results["vq_onehot"].append(onehot_target.reshape(B, T, V))
            results["vq_labels"].append(onehot_target.argmax(-1).reshape(B, T))

    if get_pred:
        results["student_logit"] = []
        results["student_prob"] = []
        for layer in self.heads:
            logit = layer(x).float()
            results["student_logit"].append(logit)
            results["student_prob"].append(F.softmax(logit / self.student_temp, dim=-1))

    return results
