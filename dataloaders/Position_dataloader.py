import numpy as np
import torch
from torch.utils.data import Dataset
from data_process.data_process_func import *
import random
import cv2

def create_sample1(image_path):

    image = load_png_image_as_array1(image_path)
    image = image.astype(np.float32)

    # Ensure the image has a batch dimension for PyTorch compatibility
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    sample = {'image': torch.from_numpy(image), 'image_path': image_path}
    # print(sample)
    return sample

class PositionDataloader1(Dataset):
    """Dataset position"""

    def __init__(self, config=None, image_path='../data/Resized_Reconstructed_Static_64_Frame.png', transform=None,
                 random_sample=True, out_size=None):
        self._iternum = config['iter_num']
        self.out_size = out_size
        self.transform = transform
        self.image_path = image_path  # Directly use the image path
        self.random_sample = random_sample

        # print("Total 1 sample")  # Only one sample since it's a single image

    def __len__(self):
        # Return the number of iterations per epoch
        return self._iternum

    def __getitem__(self, idx):
        # Load image on demand
        sample = create_sample1(self.image_path)
        # print("sample after create_sample1", sample)

        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)
            # print("sample after transformation", sample)

        return sample


class RandomDoubleCrop1(object):

    def __init__(self, output_size, foreground_only=True, small_move=False, fluct_range=[0, 0]):
        self.output_size = torch.tensor(output_size, dtype=torch.int16)
        self.img_pad = torch.div(self.output_size, 2, rounding_mode='trunc')
        self.foreground_only = foreground_only
        self.fluct_range = fluct_range  # distance, could be in pixels
        self.small_move = small_move
        # Only two dimensions are needed for 2D images
        # cor_x, cor_y = np.meshgrid(np.arange(self.output_size[1]),
        #                            np.arange(self.output_size[0]))
        # self.cor_grid = torch.from_numpy(np.concatenate((cor_x[np.newaxis],
        #                                                  cor_y[np.newaxis]), axis=0))

    def random_position(self, shape, initial_position=[0, 0], small_move=False):
        position = []
        # Start from index 1 to skip the channel dimension
        for i in range(1, len(shape)):  # This will iterate over height and width only
            if small_move:
                # Calculate the position with consideration for small_move and fluct_range
                pos_min = max(0, initial_position[i - 1] - self.fluct_range[i - 1])
                pos_max = min(shape[i] - 1, initial_position[i - 1] + self.fluct_range[i - 1])
                position.append(random.randint(pos_min, pos_max))
            else:
                # Random position without small_move considerations
                # print(shape[i])
                position.append(random.randint(0, shape[i] - 1))
        print(position)
        return torch.tensor(position, dtype=torch.int16)

    def __call__(self, sample):
        image = sample['image']  # Assuming image is a numpy array.
        nsample = {}
        nsample['image_path'] = sample['image_path']
        copied_image = image.numpy()
        copied_image = np.squeeze(copied_image)
        print(copied_image.shape)
        print(type(copied_image))
        cv2.imshow("Image", copied_image)
        cv2.waitKey(0)
        background_chosen = True
        shape_n = image.shape  # Directly using numpy shape property.
        # print("This is the shape being used", shape_n)
        random_pos0 = []
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            # Check if the randomly selected position is foreground if foreground_only is True.
            if not self.foreground_only or image[0, random_pos0[0], random_pos0[1]] >= 30:
                background_chosen = False

        half_size = np.array(self.output_size) // 2  # Assuming output_size is a tensor.
        top_left = np.maximum(random_pos0 - half_size, [0, 0])
        bottom_right = np.minimum(random_pos0 + half_size, shape_n[1:])

        # Crop the image
        cropped_image = image[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        # Determine the amount of padding needed
        padding = [(0, 0)]  # No padding for channels
        for i in range(2):
            low_pad = half_size[i] - (random_pos0[i] - top_left[i])
            high_pad = half_size[i] - (bottom_right[i] - random_pos0[i])
            padding.append((low_pad, high_pad))

        # Apply padding
        padded_image = np.pad(cropped_image, padding, mode='constant', constant_values=0)

        # Ensure the output image has the desired size
        padded_image = padded_image[:, :self.output_size[0], :self.output_size[1]]

        # print("Cropped image 0 shape: ", padded_image.shape)
        nsample['random_crop_image_0'] = padded_image
        padded_image_for_display = np.squeeze(padded_image)
        cv2.imshow('Cropped Image', padded_image_for_display)
        cv2.waitKey(1000)
        nsample['random_position_0'] = random_pos0

        # # Optional: Draw the square and midpoint on a copy of the image for visualization
        # if 'draw_square' in self.__dir__():  # Check if draw_square method exists
        #     vis_image = self.draw_square(image.copy(), random_pos0.numpy())
        #     nsample['visualized_image'] = vis_image  # Store or save this image as needed

        # Ensure random_pos0 is a tensor for this operation
        # random_pos0_tensor = torch.tensor(random_pos0, dtype=torch.float32).view(2, 1, 1)
        # nsample['random_fullsize_position_0'] = (self.cor_grid + random_pos0_tensor).numpy()

        # 2nd Crop

        background_chosen = True
        random_pos1 = []
        while background_chosen:
            random_pos1 = self.random_position(shape_n, nsample['random_position_0'], self.small_move)
            # Check the condition for the selected pixel; if it's foreground, break the loop.
            if not self.foreground_only or image[0, random_pos1[0], random_pos1[1]] >= 15:
                background_chosen = False

        top_left1 = np.maximum(random_pos1 - half_size, [0, 0])
        bottom_right1 = np.minimum(random_pos1 + half_size, shape_n[1:])

        # Crop the image
        cropped_image1 = image[:, top_left1[0]:bottom_right1[0], top_left1[1]:bottom_right1[1]]
        # Determine the amount of padding needed
        padding1 = [(0, 0)]  # No padding for channels
        for i in range(2):
            low_pad1 = half_size[i] - (random_pos1[i] - top_left1[i])
            high_pad1 = half_size[i] - (bottom_right1[i] - random_pos1[i])
            padding1.append((low_pad1, high_pad1))

        # Apply padding
        padded_image1 = np.pad(cropped_image1, padding1, mode='constant', constant_values=0)

        # Ensure the output image has the desired size
        padded_image1 = padded_image1[:, :self.output_size[0], :self.output_size[1]]
        # print("Cropped image 1 shape: ", padded_image1.shape)
        nsample['random_crop_image_1'] = padded_image1

        nsample['random_position_1'] = random_pos1

        # # Optional: Draw the square and midpoint on a copy of the image for visualization
        # if 'draw_square' in self.__dir__():  # Check if draw_square method exists
        #     vis_image1 = self.draw_square(image.copy(), random_pos1.numpy())
        #     nsample['visualized_image1'] = vis_image1  # Store or save this image as needed

        # Ensure random_pos1 is a tensor for this operation
        # random_pos1_tensor = torch.tensor(random_pos1, dtype=torch.float32).view(2, 1, 1)
        # nsample['random_fullsize_position_1'] = (self.cor_grid + random_pos1_tensor).numpy()

        key_ls = list(nsample.keys())
        for key in key_ls:
            if torch.is_tensor(nsample[key]):
                nsample[key] = nsample[key].to(torch.float32)

        return nsample

class ToPositionTensor1(object):
    """Convert ndarrays in sample to Tensors for 2D images."""

    def __call__(self, sample):

        sample['rela_distance'] = sample['random_position_0'] - sample['random_position_1']
        # sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0'] - sample['random_fullsize_position_1'])

        return sample

