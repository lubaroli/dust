import time
from pathlib import Path

import numpy as np
import torch


def get_project_root():
    return Path(__file__).parent.parent.parent


def save_progress(
    folder_name: Path = None,
    session=False,
    data=None,
    params=None,
    fig=None,
    fig_name="plot.pdf",
):
    """Saves session to the project data folder.

     Path can be specified, if not, an auto generated folder based on the
     current date-time is used. May include a plot.

    :param data: A data object to save. If None, whole session is saved.
    :type data: object
    :param folder_name: A path-like string containing the output path.
    :type folder_name: str
    :param fig: A figure object.
    :type fig: matplotlib.figure.Figure
    """
    if folder_name is None:
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    root_path = get_project_root()
    folder_path = Path(root_path, "data/local/" + str(folder_name))
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    if fig:
        plot_path = folder_path / "plots"
        if not plot_path.exists():
            plot_path.mkdir()
        try:
            fig.savefig(plot_path / fig_name)
        except AttributeError:
            pass  # fallback to matplotlib
        try:
            from matplotlib.pyplot import savefig

            savefig(plot_path / fig_name)
        except AttributeError:
            raise AttributeError("Figure does not have a save function.")

    if session is True:
        try:
            import dill
        except ImportError:
            print("Couldn't import package dill. Aborting save progress.")
            return None
        sess_path = folder_path / "session.pkl"
        dill.dump_session(sess_path)
    if data is not None:
        data_path = folder_path / "data.pkl"
        with data_path.open("wb") as fh:
            torch.save(data, fh)

    if params is not None:
        try:
            import yaml
        except ImportError:
            print("Couldn't import package PyYAML. Aborting save progress.")
            return None
        config_path = folder_path / "config.yaml"
        with config_path.open("w") as fh:
            yaml.dump(params, fh)
    return folder_path


def create_video_from_plots(save_path, plot_path=None):
    try:
        import moviepy.editor as mpy
    except ImportError:
        print("Couldn't import package MoviePy. Aborting video creation.")
        return None
    if plot_path is None:
        plot_path = save_path / "plots"
    video = mpy.ImageSequenceClip(str(plot_path), fps=20)
    video.write_videofile(str(save_path / "video.mp4"))


def to_np(x, dtype=np.float):
    if isinstance(x, list):
        return [v.cpu().numpy().astype(dtype) for v in x]
    else:
        return x.cpu().numpy().astype(dtype)


def from_np(x, dtype=torch.float):
    if isinstance(x, list):
        return [torch.from_numpy(v).type(dtype) for v in x]
    else:
        return torch.from_numpy(x).type(dtype)
