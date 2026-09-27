from torch import nn
import torch
from collections import OrderedDict
from typing import Dict, List
# Local imports
from pl_modules.base import BaseModel
from losses.womm_loss import WoMMLoss
from models.mmfusion import MMFusion

class WoMM(BaseModel):
    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(WoMM, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        loss_kwargs = dict(loss_kwargs)
        variant = loss_kwargs.pop("variant", "womm")
        if variant == "comm":
            from losses.comm_loss import CoMMLoss
            class ScaledCoMMLoss(CoMMLoss):
                def __init__(self, temperature=0.1, weights=None, reg_weight=0.5):
                    super().__init__(temperature=temperature, weights=weights)
                    self.reg_weight = reg_weight

                def forward(self, outputs):
                    out_dict = super().forward(outputs)
                    out_dict["loss"] = out_dict["loss"] * self.reg_weight
                    return out_dict
            self.loss = ScaledCoMMLoss(
                temperature=loss_kwargs.get("temperature", 0.1),
                weights=loss_kwargs.get("weights", None),
                reg_weight=loss_kwargs.get("reg_weight", 0.5)
            )
        elif variant == "womm":
            self.loss = WoMMLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss variant: {variant!r} (expected 'womm' or 'comm')")


    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))


    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        all_masks = self.gen_all_possible_masks(len(x1))
        z1 = self.encoder(x1, mask_modalities=all_masks)
        z2 = self.encoder(x2, mask_modalities=all_masks)
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        return {'aug1_embed': z1,
                'aug2_embed': z2,
                "prototype": -1}
    

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks
    
    
    def _frozen_encoder_keys(self) -> List[str]:
        """State-dict keys of the pre-trained encoders loaded frozen (`freeze=True`).

        Keyed on the flag rather than on `requires_grad`: the MultiBench transformers
        also hold non-trainable parameters (fixed sin-cos positions), and those stay in
        the checkpoint so that nothing changes for the datasets trained end to end.
        """
        return [f"encoder.encoders.{i}.{name}"
                for i, enc in enumerate(self.encoder.encoders) if getattr(enc, "freeze", False)
                for name, _ in enc.named_parameters()]

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        # Frozen pre-trained encoders are rebuilt from their released weights whenever
        # the model is instantiated, so a checkpoint only needs what training changes.
        # On MM-IMDb and Hateful Memes that drops ~1 GB per run; where the encoders are
        # trained (Trifeatures, MultiBench) the list is empty and nothing changes.
        state = checkpoint["state_dict"]
        for key in self._frozen_encoder_keys():
            state.pop(key, None)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        # Restore the dropped keys from the freshly instantiated model, so resuming
        # and `load_from_checkpoint` still load strictly.
        state, own = checkpoint["state_dict"], self.state_dict()
        for key in self._frozen_encoder_keys():
            if key not in state:
                state[key] = own[key]

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal features from the encoder.
           Args:
                loader: Dataset loader to serve `(X, y)` tuples.
                kwargs: given to `encoder.forward()`
           Returns: 
                Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor): # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                if isinstance(output, list):
                    output = output[0]
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)
