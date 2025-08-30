import numpy as np
import stim
import sinter

from .dem_to_matrices import detector_error_model_to_check_matrices
from .qecdec import BPDecoder, DMemBPDecoder


class _Sinter_CompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, decoder, num_detectors: int, obsmat: np.ndarray):
        self.decoder = decoder
        self.num_detectors = num_detectors
        self.obsmat = obsmat

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        syndrome_batch = np.unpackbits(
            bit_packed_detection_event_data, axis=1, bitorder="little")[:, :self.num_detectors]
        ehat = self.decoder.decode_batch(syndrome_batch)
        observable_predict = (ehat @ self.obsmat.T) % 2

        bit_packed_observable_predict = np.packbits(
            observable_predict, axis=1, bitorder="little")
        return bit_packed_observable_predict


class Sinter_BPDecoder(sinter.Decoder):
    def __init__(self, *, max_iter: int, scaling_factor: float | None = None):
        self.max_iter = max_iter
        self.scaling_factor = scaling_factor

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> _Sinter_CompiledDecoder:
        matrices = detector_error_model_to_check_matrices(dem)
        chkmat, obsmat, pvec = matrices.check_matrix, matrices.observables_matrix, matrices.priors
        chkmat = chkmat.toarray().astype(np.uint8)
        obsmat = obsmat.toarray().astype(np.uint8)
        pvec = pvec.astype(np.float64)

        return _Sinter_CompiledDecoder(
            decoder=BPDecoder(
                chkmat, pvec,
                max_iter=self.max_iter,
                scaling_factor=self.scaling_factor,
            ),
            num_detectors=chkmat.shape[0],
            obsmat=obsmat
        )


class Sinter_DMemBPDecoder(sinter.Decoder):
    def __init__(
        self,
        *,
        max_iter: int,
        gamma: np.ndarray,
        scaling_factor: float | None = None
    ):
        self.max_iter = max_iter
        self.scaling_factor = scaling_factor
        self.gamma = gamma

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> _Sinter_CompiledDecoder:
        matrices = detector_error_model_to_check_matrices(dem)
        chkmat, obsmat, pvec = matrices.check_matrix, matrices.observables_matrix, matrices.priors
        chkmat = chkmat.toarray().astype(np.uint8)
        obsmat = obsmat.toarray().astype(np.uint8)
        pvec = pvec.astype(np.float64)

        num_detectors, num_error_mechanisms = chkmat.shape

        return _Sinter_CompiledDecoder(
            decoder=DMemBPDecoder(
                chkmat, pvec,
                gamma=self.gamma,
                max_iter=self.max_iter,
                scaling_factor=self.scaling_factor,
            ),
            num_detectors=num_detectors,
            obsmat=obsmat
        )


__all__ = ["Sinter_BPDecoder", "Sinter_DMemBPDecoder"]
