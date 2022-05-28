# adapted from https://medium.datadriveninvestor.com/xai-with-lime-for-cnn-models-5560a486578
import torch
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def run_model(model, x):
    """
    Takes numpy images in format batch x H x W x 3
    and runs model inference
    """
    x = torch.tensor(x).permute(0, 3, 1, 2)  # swap to batch x 3 x H x W
    return model(x).detach().numpy()


def pred_single_img(model, x):
    """
    Takes numpy image in format H x W x 3
    and outputs predicted label
    """
    return run_model(model, x.reshape((1, *x.shape))).argmax(axis=1)[0]


def get_lime_explanation(model, image):
    """
    Takes pytorch tensor in format 3 x H x W
    and return prediction and lime explanation
    """
    model.eval()
    image = image.permute(1, 2, 0).numpy()
    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image,
        lambda x: run_model(model, x)
    )
    pred = pred_single_img(model, image)
    image, mask = explanation.get_image_and_mask(
        pred,
        positive_only=True,
        hide_rest=False)
    return pred, mark_boundaries(image, mask)


def save_lime(np_image, fname):
    dpi = 1000.
    res = 128.
    fig = plt.figure(dpi=dpi, tight_layout=True,
                     frameon=False, figsize=(res/dpi, res/dpi))
    fig.figimage(np_image)
    plt.savefig(fname)
    plt.close(fig)
