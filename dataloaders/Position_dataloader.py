import torch
from torch.utils.data import Dataset
from data_process.data_process_func import *
from multiprocessing import Pool, cpu_count

def create_sample1(image_path, out_size):
    """
    Create a sample dictionary with image data loaded from a JPEG file.

    Args:
        image_path (str): Path to the JPEG image.
        out_size (tuple): Desired output size as (height, width).

    Returns:
        dict: A dictionary containing the image tensor and additional information.
    """
    image = load_jpeg_image_as_array1(image_path)
    image = image.astype(np.float32)

    # If the image is grayscale, it will be 2D. If RGB, it will be 3D.
    if len(image.shape) == 2:  # Grayscale
        image = np.expand_dims(image, axis=-1)  # Add a channel dimension

    if out_size:
        if image.shape[0] <= out_size[0] or image.shape[1] <= out_size[1]:
            ph = max((out_size[0] - image.shape[0]) // 2 + 3, 0)
            pw = max((out_size[1] - image.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(ph, ph), (pw, pw), (0, 0)], mode='constant', constant_values=0)

    # Ensure the image has a batch dimension for PyTorch compatibility
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    sample = {'image': torch.from_numpy(image), 'image_path': image_path}
    return sample

class PositionDataloader1(Dataset):
    """ Dataset position """

    def __init__(self, config=None, image_name_list='train', num=None, transform=None,
                 random_sample=True, load_aug=False, load_memory=True, out_size=None):
        self._iternum = config['iter_num']
        self.out_size = out_size
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        image_task_dic = {}
        self.iternum = 0
        self.load_aug = load_aug
        self.load_memory = load_memory
        self.random_sample = random_sample
        self.image_name_list = read_file_list(image_name_list)

        if load_memory:
            # p = Pool(2)
            p = Pool(cpu_count())
            for image_name in self.image_name_list:
                image_task_dic[image_name] = p.apply_async(create_sample1, args=(image_name, out_size,))
            p.close()
            p.join()
            for image_name in image_task_dic.keys():
                self.image_dic[image_name] = image_task_dic[image_name].get()

        if num is not None:
            self.image_name_list = self.image_name_list[:num]
        print("total {} samples".format(len(self.image_name_list)))

    def __len__(self):
        if self.random_sample:
            return self._iternum
        else:
            return len(self.image_name_list)

    def __getitem__(self, idx):
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_name_list, 1)[0]].copy()
        else:
            if self.random_sample:
                image_name = random.sample(self.image_name_list, 1)[0]
            else:
                image_name = self.image_name_list[idx]
            sample = create_sample1(image_name, self.out_size)
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomDoubleCrop1(object):
    """
    Randomly crop several images in one sample;
    distance is a vector (could be positive or negative), representing the vector
    from image1 to image2.
    Args:
    output_size (tuple): Desired output size (height, width)
    """

    def __init__(self, output_size, foreground_only=True, small_move=False, fluct_range=[0, 0]):
        self.output_size = torch.tensor(output_size, dtype=torch.int16)
        self.img_pad = torch.div(self.output_size, 2, rounding_mode='trunc')
        self.foreground_only = foreground_only
        self.fluct_range = fluct_range  # distance, could be in pixels
        self.small_move = small_move
        # Only two dimensions are needed for 2D images
        cor_x, cor_y = np.meshgrid(np.arange(self.output_size[1]),
                                   np.arange(self.output_size[0]))
        self.cor_grid = torch.from_numpy(np.concatenate((cor_x[np.newaxis],
                                                         cor_y[np.newaxis]), axis=0))

    def random_position(self, shape, initial_position=[0, 0], small_move=False):
        position = []
        for i in range(len(shape)):  # This will now iterate twice (for x and y)
            if small_move:
                # Calculate the position with consideration for small_move and fluct_range
                pos_min = max(0, initial_position[i] - self.fluct_range[i])
                pos_max = min(shape[i] - 1, initial_position[i] + self.fluct_range[i])
                position.append(random.randint(pos_min, pos_max))
            else:
                # Random position without small_move considerations
                position.append(random.randint(0, shape[i] - 1))
        return torch.tensor(position, dtype=torch.int16)

    def __call__(self, sample):
        image = sample['image']  # Assuming image is a numpy array.
        nsample = {}
        nsample['image_path'] = sample['image_path']

        background_chosen = True
        shape_n = image.shape  # Directly using numpy shape property.
        while background_chosen:
            random_pos0 = self.random_position(shape_n)
            # Check if the randomly selected position is foreground if foreground_only is True.
            if not self.foreground_only or image[random_pos0[1], random_pos0[0]] >= 45:
                background_chosen = False

        half_size = (self.output_size // 2).numpy()  # Assuming output_size is a tensor.
        top_left = np.maximum(random_pos0 - half_size, 0)
        bottom_right = np.minimum(random_pos0 + half_size, shape_n)

        # Crop the image
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate padding
        pad_width = ((abs(min(0, top_left[1])), max(0, bottom_right[1] - shape_n[0])),
                     (abs(min(0, top_left[0])), max(0, bottom_right[0] - shape_n[1])))

        # Pad the image
        padded_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
        nsample['random_crop_image_0'] = padded_image

        nsample['random_position_0'] = random_pos0

        # # Optional: Draw the square and midpoint on a copy of the image for visualization
        # if 'draw_square' in self.__dir__():  # Check if draw_square method exists
        #     vis_image = self.draw_square(image.copy(), random_pos0.numpy())
        #     nsample['visualized_image'] = vis_image  # Store or save this image as needed

        # Ensure random_pos0 is a tensor for this operation
        random_pos0_tensor = torch.tensor(random_pos0, dtype=torch.float32).view(2, 1, 1)
        nsample['random_fullsize_position_0'] = (self.cor_grid + random_pos0_tensor).numpy()

        # 2nd Crop

        background_chosen = True
        while background_chosen:
            random_pos1 = self.random_position(shape_n, nsample['random_position_0'], self.small_move)
            # Check the condition for the selected pixel; if it's foreground, break the loop.
            if not self.foregroung_only or image[random_pos1[0], random_pos1[1]] >= 45:
                background_chosen = False

        top_left1 = np.maximum(random_pos1 - half_size, 0)
        bottom_right1 = np.minimum(random_pos1 + half_size, shape_n)

        # Crop the image
        cropped_image1 = image[top_left1[1]:bottom_right1[1], top_left1[0]:bottom_right1[0]]

        # Calculate padding
        pad_width1 = ((abs(min(0, top_left1[1])), max(0, bottom_right1[1] - shape_n[0])),
                     (abs(min(0, top_left1[0])), max(0, bottom_right1[0] - shape_n[1])))

        # Pad the image
        padded_image1 = np.pad(cropped_image1, pad_width1, mode='constant', constant_values=0)
        nsample['random_crop_image_1'] = padded_image1

        nsample['random_position_1'] = random_pos1

        # # Optional: Draw the square and midpoint on a copy of the image for visualization
        # if 'draw_square' in self.__dir__():  # Check if draw_square method exists
        #     vis_image1 = self.draw_square(image.copy(), random_pos1.numpy())
        #     nsample['visualized_image1'] = vis_image1  # Store or save this image as needed

        # Ensure random_pos1 is a tensor for this operation
        random_pos1_tensor = torch.tensor(random_pos1, dtype=torch.float32).view(2, 1, 1)
        nsample['random_fullsize_position_1'] = (self.cor_grid + random_pos1_tensor).numpy()

        key_ls = list(nsample.keys())
        for key in key_ls:
            if torch.is_tensor(nsample[key]):
                nsample[key] = nsample[key].to(torch.float32)

        return nsample

class RandomDoubleMask1(object):
    def __init__(self, max_round=1, include_0=['random_crop_image_0'], include_1=['random_crop_image_1'],
                 mask_size=[0, 0]):
        self.include_0 = include_0
        self.include_1 = include_1
        self.max_round = max_round
        self.mask_size = mask_size  # Should now be [height, width]

    def __call__(self, sample):
        for include in self.include_0[0], self.include_1[0]:
            max_round = np.random.randint(1, self.max_round + 1)
            sample[include.replace('random', 'random_mask')] = torch.clone(sample[include])
            mask_image = sample[include].clone()
            shape = mask_image.shape[1:]  # Shape is now just height and width

            for _ in range(max_round):
                min_cor = []
                for i in range(2):  # Loop over 2 dimensions instead of 3
                    min_cor.append(np.random.randint(0, shape[i] - self.mask_size[i]))

                # Apply the mask
                sample[include.replace('random', 'random_mask')][:,
                min_cor[0]:min_cor[0] + self.mask_size[0], min_cor[1]:min_cor[1] + self.mask_size[1]] = 0

        return sample

class ToPositionTensor1(object):
    """Convert ndarrays in sample to Tensors for 2D images."""

    def __call__(self, sample):

        sample['rela_distance'] = sample['random_position_0'] - sample['random_position_1']
        sample['rela_fullsize_distance'] = (sample['random_fullsize_position_0'] - sample['random_fullsize_position_1'])

        return sample

