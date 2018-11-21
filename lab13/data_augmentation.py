from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from matplotlib import pyplot as plt
import cv2

input_path = 'data/logo.png'
output_path = 'data_augmented/dog_aug{}.jpg'
count = 9

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# load image to array
image = img_to_array(load_img(input_path))

# reshape to array rank 4
image = image.reshape((1,) + image.shape)

# let's create infinite flow of images
images_flow = gen.flow(image, batch_size=1)
for i, new_images in enumerate(images_flow):
    i += 1
    # we access only first image because of batch_size=1
    new_image = array_to_img(new_images[0], scale=True)
    new_image.save(output_path.format(i))
    if i >= count:
        break

# plot augmented data
f, axarr = plt.subplots(3, 3, figsize=(20, 20))
k = 1
for i in range(3):
    for j in range(3):
        img = cv2.imread("data_augmented/dog_aug{}.jpg".format(k))
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        k += 1
        axarr[i, j].imshow(rgb_img)
        axarr[i, j].xaxis.set_major_locator(plt.NullLocator())
        axarr[i, j].yaxis.set_major_locator(plt.NullLocator())
plt.show()
