"""Bridge PyTorch decoder models to sinter for Monte Carlo benchmarking."""
import time
import sys

import numpy as np
import sinter
import torch

from .models import DecoderModel
from .utils.llr_utils import llrs_to_ehat
from .utils.tensor_utils import INT_DTYPE, FLOAT_DTYPE


class PyTorchSinterDecoder(sinter.Decoder):
    """Wrap a `DecoderModel` as a `sinter.Decoder` for use in `sinter.collect`."""
    PROFILING = False
    PROFILING_INTERVAL = 50

    def __init__(
        self,
        model: DecoderModel,
        obsmat: np.ndarray,
        *,
        device: str,
        bypass: bool,
    ):
        """
        Parameters
        ----------
        model : DecoderModel
            PyTorch decoder model.

        obsmat : np.ndarray
            Observable matrix, shape=(num_obsers, num_vars). Used to convert error predictions to 
            observable predictions for sinter.

        device : str
            Device for inference ("cuda", "cpu", etc.).

        bypass : bool
            If True, always return all-zero predictions for shots with all-zero syndrome.
        """
        self.model = model
        self.obsmat = obsmat
        self.num_chks = model.num_chks
        self.device = device
        self.bypass = bypass

        if device == "cpu":
            self._chkmat_tensor = torch.tensor(model.pcm, dtype=INT_DTYPE, device=device)
        else:
            self._chkmat_tensor = torch.tensor(model.pcm, dtype=FLOAT_DTYPE, device=device)

        # Move model to device.
        self.model.to(device)
        # Set model to evaluation mode.
        self.model.eval()

        # TODO: Add option to compile model with torch.compile.
        # if hasattr(torch, 'compile'):
        #     print("Compiling model with torch.compile...")
        #     # torch.set_float32_matmul_precision('high')
        #     self.model = torch.compile(self.model, mode="reduce-overhead")

        if self.PROFILING:
            self._calls = 0
            self._batch_sizes = []
            self._timing = {"preprocess": 0.0, "model": 0.0, "llr_to_ehat": 0.0, "postprocess": 0.0}

    def compile_decoder_for_dem(self, *, dem):
        """Dummy method to satisfy the interface of `sinter.Decoder`. You should never call this method."""
        return self

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        """
        Decode bit-packed syndromes, and return bit-packed observable predictions.
        This method is meant to be used by `sinter.collect`. You should not call this method directly.
        """
        if self.PROFILING:
            t0 = time.perf_counter()

        unpacked = np.unpackbits(bit_packed_detection_event_data, axis=1, bitorder="little")
        unpacked = unpacked[:, :self.num_chks]
        syndromes = torch.from_numpy(unpacked).to(dtype=INT_DTYPE, device=self.device)

        if self.PROFILING:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        with torch.no_grad():
            llrs = self.model(syndromes)

            if self.PROFILING:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

            ehat, _, _ = llrs_to_ehat(llrs, syndromes, self._chkmat_tensor)

            if self.PROFILING:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

        ehat_np = ehat.cpu().numpy().astype(np.uint8)

        if self.bypass:
            mask = np.all(unpacked == 0, axis=1)
            ehat_np[mask] = 0

        observable_predict = (ehat_np @ self.obsmat.T) % 2
        result = np.packbits(observable_predict, axis=1, bitorder="little")

        if self.PROFILING:
            t4 = time.perf_counter()

            self._timing["preprocess"] += t1 - t0
            self._timing["model"] += t2 - t1
            self._timing["llr_to_ehat"] += t3 - t2
            self._timing["postprocess"] += t4 - t3
            self._calls += 1
            self._batch_sizes.append(syndromes.size(0))

            if self._calls % self.PROFILING_INTERVAL == 0:
                total_time = sum(self._timing.values())
                print("[Batch sizes]\n"
                      f"min={np.min(self._batch_sizes)} | "
                      f"max={np.max(self._batch_sizes)} | "
                      f"mean={np.mean(self._batch_sizes):.2f} | "
                      f"std={np.std(self._batch_sizes):.2f}")
                print(f"[Time for {self.PROFILING_INTERVAL} batches]\n"
                      f"preprocess={self._timing['preprocess']:.3f}s | "
                      f"model={self._timing['model']:.3f}s | "
                      f"llr_to_ehat={self._timing['llr_to_ehat']:.3f}s | "
                      f"postprocess={self._timing['postprocess']:.3f}s | "
                      f"total={total_time:.3f}s")
                sys.stdout.flush()
                self._timing = {"preprocess": 0.0, "model": 0.0, "llr_to_ehat": 0.0, "postprocess": 0.0}
                self._batch_sizes = []

        return result
