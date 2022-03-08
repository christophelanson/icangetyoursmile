from PIL import Image, ImageOps

def preproc_image(path, angle=90, image_size=(64,64)):
    """
    Enter a path to an image.
    The image is opened, rotated and resized according to parameters.

    """
    # Open Image
    im = Image.open(path)

    # Rotate Image
    im_rotated = im.rotate(angle=angle, expand=True)

    # Resize Image
    im_final = ImageOps.fit(im_rotated, size=image_size, bleed=0.0, centering=(0.5,0.5))

    return im_final
