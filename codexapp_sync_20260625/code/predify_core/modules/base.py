import torch
import torch.nn as nn


def _is_zero_multiplier(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool((value.detach() == 0).all().item())
    return float(value) == 0.0


def _tensor_rms(value: torch.Tensor) -> float:
    if value is None:
        return 0.0
    detached = value.detach().float()
    return float(torch.sqrt(torch.mean(detached.pow(2))).item())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


class Predictor(nn.Module):
    r"""
    Implements a predictor module. Predictors generate predictions via the given `pmodule` from input representations.
    A Predictor contains three main attributes:

    * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
    * :attr:`rep` is the output representation of the Predictor. It is equal to the input representation.
    * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).

    Args:
        pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
        random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)
    """

    def __init__(self, pmodule: nn.Module, random_init: bool):
        super(Predictor, self).__init__()
        self.random_init  = random_init     # indicates if in timestep 0 the feedback is random
        
        ### Modules
        self.pmodule = pmodule    # feedback module

        ### Memory
        self.rep = None           # last representation
        self.prd = None           # last prediction

    def forward(self, ff: torch.Tensor, build_graph: bool=False):
        r"""
        Computes the prediction from the given representation.

        Args:
            ff (torch.Tensor): Input representation.
            build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)

        Returns:
            Tuple: the output representation and prediction
        """
        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            self.rep = ff

        with torch.enable_grad():
            if not self.rep.requires_grad:
                self.rep.requires_grad = True 
        
            self.prd = self.pmodule(self.rep)
        
            if not build_graph:
                self.prd = self.prd.detach()

        return self.rep, self.prd


    def reset(self):
        r"""
        To be called for each new batch of images.
        """
        self.rep = None           # last representation
        self.prd = None           # last prediction

class PCoder(Predictor):
    r"""
    Implements a predictive coding module. PCoders generate predictions via the given `pmodule` from input representations while generating their own representation
    with the following equation:
    
    .. math::
        \beta(feedforward) + \lambda(feedback) + (1-\beta-\lambda)(memory) - \alpha(gradient)
    
    where :math:`0\leq \beta + \lambda \leq 1`.

    A PCoder contains the following attributes:

    * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
    * :attr:`rep` is the output representation of the Predictor which updates based on the equation above.
    * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).
    * :attr:`grd` is the gradient of the prediction error.

    Args:
        pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
        has_feedback (boolean): Indicates whether the module receives feedback or not.
        random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, random_init)

        self.has_feedback = has_feedback    # indicates if the modules receives feedback

        ### Memory
        self.grd = None           # last error gradient

    def forward(self, ff: torch.Tensor, fb: torch.Tensor, target: torch.Tensor, build_graph: bool=False, ffm: float=None, fbm: float=None, erm: float=None):
        r"""
        Updates PCoder for one timestep.

        Args:
            ff (torch.Tensor): Feedforward drive.
            fb (torch.Tensor): Feedback drive (`None` if `has_feedback` is `False`).
            target (torch.Tensor): Target representation to compare with the prediction.
            build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
            ffm (float): The value of :math:`\beta`.
            fbm (float): The value of :math:`\lambda`.
            erm (float): The value of :math:`\alpha`.
        
        Returns:
            Tuple: the output representation and prediction
        """

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            if self.has_feedback:
                self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*self.grd
            else:
                self.rep = ffm*ff + (1-ffm)*self.rep - erm*self.grd

        skip_error_gradient = (not build_graph) and _is_zero_multiplier(erm)
        if skip_error_gradient:
            with torch.no_grad():
                self.prd = self.pmodule(self.rep)
                self.grd = torch.zeros_like(self.rep)
                self.prediction_error = torch.zeros((), device=self.rep.device, dtype=self.rep.dtype)
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
        else:
            with torch.enable_grad():
                if not self.rep.requires_grad:
                    self.rep.requires_grad = True 
        
                self.prd = self.pmodule(self.rep)
                self.prediction_error =  nn.functional.mse_loss(self.prd, target)
                self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]
                
                if not build_graph:
                    self.prd = self.prd.detach()
                    self.grd = self.grd.detach()
                    self.rep = self.rep.detach()
                    self.prediction_error = self.prediction_error.detach()
        return self.rep, self.prd

    def reset(self):
        r"""
        To be called for each new batch of images.
        """
        super().reset()
        self.grd = None           # last error gradient

class PCoderN(PCoder):
    r"""
    Implements a predictive coding module with scaled gradient term.
    PCoderNs generate predictions via the given `pmodule` from input representations while correcting their own representation
    with the following equation:
    
    .. math::
        \beta(feedforward) + \lambda(feedback) + (1-\beta-\lambda)(memory) - \alpha(gradient)
    
    where :math:`0\leq \beta + \lambda \leq 1`.

    A PCoder contains the following attributes:

    * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
    * :attr:`rep` is the output representation of the Predictor which updates based on the equation above.
    * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).
    * :attr:`grd` is the gradient of the prediction error.

    Args:
        pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
        has_feedback (boolean): Indicates whether the module receives feedback or not.
        random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)

    Note:
        K-C normalization is used for the gradient term, where K is the size of the prediction tensor and C is the effective window size of a predictor cell.
        C is computed in this way:
            - generate random input X
            - pass it through the prediction module and get the output P
            - repeat 10 times:
                - make a copy of X as X' and change the central cell randomly
                - pass it through the prediction module and compute the output P'
                - count number of different cells between P and P'
            - use the averge "difference" as the C
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, has_feedback, random_init)
        self.register_buffer('C_sqrt', torch.tensor(-1, dtype=torch.float))
        self.last_update_stats = None

    def _record_update_stats(
        self,
        prev_rep: torch.Tensor,
        ff_delta: torch.Tensor,
        fb_delta: torch.Tensor,
        alpha_term: torch.Tensor,
        grad: torch.Tensor,
        error_scale,
        erm,
        source_prediction_error,
    ):
        drive_delta = ff_delta + fb_delta
        total_delta = self.rep - prev_rep

        ff_delta_rms = _tensor_rms(ff_delta)
        fb_delta_rms = _tensor_rms(fb_delta)
        drive_delta_rms = _tensor_rms(drive_delta)
        alpha_delta_rms = _tensor_rms(alpha_term)
        total_delta_rms = _tensor_rms(total_delta)
        grad_rms = _tensor_rms(grad)

        self.last_update_stats = {
            "has_previous_state": True,
            "ff_delta_rms": ff_delta_rms,
            "fb_delta_rms": fb_delta_rms,
            "drive_delta_rms": drive_delta_rms,
            "alpha_delta_rms": alpha_delta_rms,
            "total_delta_rms": total_delta_rms,
            "grad_rms": grad_rms,
            "alpha_to_drive_ratio": _safe_ratio(alpha_delta_rms, drive_delta_rms),
            "alpha_to_total_ratio": _safe_ratio(alpha_delta_rms, total_delta_rms),
            "error_scale": float(error_scale),
            "erm": float(erm.detach().item()) if isinstance(erm, torch.Tensor) else float(erm),
            "source_prediction_error": float(source_prediction_error),
        }

    def _record_init_stats(self):
        self.last_update_stats = {
            "has_previous_state": False,
            "ff_delta_rms": 0.0,
            "fb_delta_rms": 0.0,
            "drive_delta_rms": 0.0,
            "alpha_delta_rms": 0.0,
            "total_delta_rms": 0.0,
            "grad_rms": 0.0,
            "alpha_to_drive_ratio": 0.0,
            "alpha_to_total_ratio": 0.0,
            "error_scale": 0.0,
            "erm": 0.0,
            "source_prediction_error": 0.0,
        }
        
    
    def compute_C_sqrt(self, target):
        r"""
        Computes `C` and returns its square root.
        `target` is the tensor to compare the prediction with
        """
        if self.rep is None:
            raise Exception("PCoder's representation cannot be `None` while executing this function.")

        x = self.rep.detach().clone()
        x.requires_grad = True
        with torch.enable_grad():
            xpred = self.pmodule(x)
            # xloss = nn.functional.mse_loss(xpred, target)
            # xgrad = torch.autograd.grad(xloss, x, retain_graph=True)[0]

        xpred_orig = xpred.detach().clone()

        cnt = 0
        for repeat in range(10):
            x = self.rep.detach().clone()
            
            x[:,x.shape[1]//2,x.shape[2]//2,x.shape[3]//2] = torch.randint(-10000,10000,(x.shape[0],), device=x.device).float()
            x.requires_grad = True
            with torch.enable_grad():
                xpred = self.pmodule(x)
                # xloss = nn.functional.mse_loss(xpred, target)
                # xgrad = torch.autograd.grad(xloss, x, retain_graph=True)[0]
            
            xpred_rand = xpred.detach().clone()
            with torch.no_grad():
                
                diff = xpred_orig - xpred_rand

                cnt += (xpred_orig != xpred_rand).sum().float() / diff.shape[0]   # divided by the batch size
        cnt = cnt / 10.0
        self.C_sqrt = torch.sqrt(cnt)

    def forward(self, ff: torch.Tensor, fb: torch.Tensor, target: torch.Tensor, build_graph: bool=False, ffm: float=None, fbm: float=None, erm: float=None):
        r"""
        Updates PCoder for one timestep.

        Args:
            ff (torch.Tensor): Feedforward drive.
            fb (torch.Tensor): Feedback drive (`None` if `has_feedback` is `False`).
            target (torch.Tensor): Target representation to compare with the prediction.
            build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
            ffm (float): The value of :math:`\beta`.
            fbm (float): The value of :math:`\lambda`.
            erm (float): The value of :math:`\alpha`.
        
        Returns:
            Tuple: the output representation and prediction
        """

        skip_error_gradient = (not build_graph) and _is_zero_multiplier(erm)

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
            self._record_init_stats()
        else:
            prev_rep = self.rep
            if self.has_feedback:
                ff_delta = ffm * (ff - prev_rep)
                fb_delta = fbm * (fb - prev_rep)
            else:
                ff_delta = ffm * (ff - prev_rep)
                fb_delta = torch.zeros_like(prev_rep)

            alpha_term = torch.zeros_like(prev_rep)
            error_scale = 0.0
            source_prediction_error = 0.0 if getattr(self, "prediction_error", None) is None else float(
                self.prediction_error.detach().item()
            )
            if not skip_error_gradient:
                error_scale = self.prd.numel()/self.C_sqrt
                alpha_term = erm * error_scale * self.grd

            self.rep = prev_rep + ff_delta + fb_delta - alpha_term
            self._record_update_stats(
                prev_rep=prev_rep,
                ff_delta=ff_delta,
                fb_delta=fb_delta,
                alpha_term=alpha_term,
                grad=self.grd if self.grd is not None else torch.zeros_like(prev_rep),
                error_scale=error_scale,
                erm=erm,
                source_prediction_error=source_prediction_error,
            )

        if self.C_sqrt == -1 and not skip_error_gradient:
            self.compute_C_sqrt(target)
            # print(self.C_sqrt * self.C_sqrt)

        if skip_error_gradient:
            with torch.no_grad():
                self.prd = self.pmodule(self.rep)
                self.grd = torch.zeros_like(self.rep)
                self.prediction_error = torch.zeros((), device=self.rep.device, dtype=self.rep.dtype)
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
        else:
            with torch.enable_grad():
                if not self.rep.requires_grad:
                    self.rep.requires_grad = True 
        
                self.prd = self.pmodule(self.rep)
                self.prediction_error  = nn.functional.mse_loss(self.prd, target)
                self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]
                
                if not build_graph:
                    self.prd = self.prd.detach()
                    self.grd = self.grd.detach()
                    self.rep = self.rep.detach()
                    self.prediction_error = self.prediction_error.detach()

        return self.rep, self.prd

    def reset(self):
        super().reset()
        self.last_update_stats = None


class DynamicErrorPCoderN(PCoderN):
    r"""
    PCoderN variant that replaces instantaneous MSE error with a discrete
    dynamic error state:

    .. math::
        \epsilon^{k+1} = \frac{T_s}{\tau}(target - prediction)
            + (1 - \frac{T_s}{\tau})\epsilon^k

    The state correction still uses a gradient projected back to ``rep`` so
    the public PCoder interface and tensor shapes remain compatible with the
    existing Predify network wrappers.
    """

    def __init__(
        self,
        pmodule: nn.Module,
        has_feedback: bool,
        random_init: bool,
        sample_time: float = 0.03,
        tau: float = 0.05,
    ):
        super().__init__(pmodule, has_feedback, random_init)
        if sample_time <= 0:
            raise ValueError("sample_time must be positive.")
        if tau <= 0:
            raise ValueError("tau must be positive.")
        self.register_buffer("sample_time", torch.tensor(float(sample_time), dtype=torch.float))
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float))
        self.dynamic_error = None

    def _dynamic_error_energy(self, target: torch.Tensor, build_graph: bool):
        residual = target - self.prd
        if self.dynamic_error is None or self.dynamic_error.shape != residual.shape:
            previous_error = torch.zeros_like(residual)
        else:
            previous_error = self.dynamic_error if build_graph else self.dynamic_error.detach()

        gamma = (self.sample_time / self.tau).to(device=residual.device, dtype=residual.dtype)
        self.dynamic_error = gamma * residual + (1.0 - gamma) * previous_error
        return torch.mean(self.dynamic_error.pow(2))

    def forward(self, ff: torch.Tensor, fb: torch.Tensor, target: torch.Tensor, build_graph: bool=False, ffm: float=None, fbm: float=None, erm: float=None):
        skip_error_gradient = (not build_graph) and _is_zero_multiplier(erm)

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
            self._record_init_stats()
        else:
            prev_rep = self.rep
            if self.has_feedback:
                ff_delta = ffm * (ff - prev_rep)
                fb_delta = fbm * (fb - prev_rep)
            else:
                ff_delta = ffm * (ff - prev_rep)
                fb_delta = torch.zeros_like(prev_rep)

            alpha_term = torch.zeros_like(prev_rep)
            error_scale = 0.0
            source_prediction_error = 0.0 if getattr(self, "prediction_error", None) is None else float(
                self.prediction_error.detach().item()
            )
            if not skip_error_gradient:
                error_scale = self.prd.numel()/self.C_sqrt
                alpha_term = erm * error_scale * self.grd

            self.rep = prev_rep + ff_delta + fb_delta - alpha_term
            self._record_update_stats(
                prev_rep=prev_rep,
                ff_delta=ff_delta,
                fb_delta=fb_delta,
                alpha_term=alpha_term,
                grad=self.grd if self.grd is not None else torch.zeros_like(prev_rep),
                error_scale=error_scale,
                erm=erm,
                source_prediction_error=source_prediction_error,
            )

        if self.C_sqrt == -1 and not skip_error_gradient:
            self.compute_C_sqrt(target)

        if skip_error_gradient:
            with torch.no_grad():
                self.prd = self.pmodule(self.rep)
                self.grd = torch.zeros_like(self.rep)
                self.prediction_error = torch.zeros((), device=self.rep.device, dtype=self.rep.dtype)
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
        else:
            with torch.enable_grad():
                if not self.rep.requires_grad:
                    self.rep.requires_grad = True

                self.prd = self.pmodule(self.rep)
                self.prediction_error = self._dynamic_error_energy(target, build_graph)
                self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]

                if not build_graph:
                    self.prd = self.prd.detach()
                    self.grd = self.grd.detach()
                    self.rep = self.rep.detach()
                    self.prediction_error = self.prediction_error.detach()
                    self.dynamic_error = self.dynamic_error.detach()

        return self.rep, self.prd

    def reset(self):
        super().reset()
        self.dynamic_error = None
