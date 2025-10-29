
import numpy as np
def metrics(Y_true, Y_pred):
    Yt = np.atleast_2d(Y_true)
    Yp = np.atleast_2d(Y_pred)
    if Yt.shape != Yp.shape:
        try:
            Yp = Yp.reshape(Yt.shape)
        except Exception:
            pass
    N = Yt.shape[0]
    diffs = np.abs(Yt - Yp)
    sq = (Yt - Yp) ** 2
    EG_per = np.sum(diffs, axis=0) / N
    MAE_per = np.mean(diffs, axis=0)
    RMSE_per = np.sqrt(np.mean(sq, axis=0))
    return {
        "EG_per_output": EG_per.tolist(),
        "MAE_per_output": MAE_per.tolist(),
        "RMSE_per_output": RMSE_per.tolist(),
        "EG": float(np.mean(EG_per)),
        "MAE": float(np.mean(MAE_per)),
        "RMSE": float(np.mean(RMSE_per)),
        "N": int(N)
    }
