# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#pylint:disable=unused-argument
"""
Various helpful augmentation functions.

These are intended to be included in train: augmentations in a yaml file.
See the `delta.config` documentation for details.
"""
import math

import tensorflow as tf

from delta.config.extensions import register_augmentation


def _image_projective_transform(image: tf.Tensor,
                                transform: tf.Tensor,
                                *,
                                fill_mode: str = 'REFLECT',
                                interpolation: str = 'BILINEAR',
                                fill_value: float = 0.0) -> tf.Tensor:
    """Apply a projective transform to a single image tensor."""
    image = tf.convert_to_tensor(image)
    original_dtype = image.dtype

    squeeze_channel = False
    if image.shape.rank == 2:
        image = image[..., tf.newaxis]
        squeeze_channel = True

    if image.shape.rank != 3:
        raise ValueError('Expected image tensor with rank 2 or 3.')

    image = tf.cast(image, tf.float32)
    batched_image = tf.expand_dims(image, 0)
    batched_transform = tf.reshape(tf.cast(transform, tf.float32), [1, 8])

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=batched_image,
        transforms=batched_transform,
        output_shape=tf.shape(batched_image)[1:3],
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
        fill_value=tf.cast(fill_value, batched_image.dtype))

    result = transformed[0]
    if squeeze_channel:
        result = result[..., 0]

    return tf.cast(result, original_dtype)


def _rotation_transform(angle: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """Compute the projective transform for a rotation about the image center."""
    angle = tf.cast(angle, tf.float32)
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    cos_theta = tf.math.cos(angle)
    sin_theta = tf.math.sin(angle)

    height_minus_one = height - 1.0
    width_minus_one = width - 1.0

    x_offset = ((width_minus_one) - (cos_theta * width_minus_one - sin_theta * height_minus_one)) / 2.0
    y_offset = ((height_minus_one) - (sin_theta * width_minus_one + cos_theta * height_minus_one)) / 2.0

    return tf.stack([
        cos_theta,
        -sin_theta,
        x_offset,
        sin_theta,
        cos_theta,
        y_offset,
        tf.zeros([], tf.float32),
        tf.zeros([], tf.float32)
    ])


def _translation_transform(offsets: tf.Tensor) -> tf.Tensor:
    """Compute the projective transform for a translation."""
    offsets = tf.cast(offsets, tf.float32)
    return tf.stack([
        1.0,
        0.0,
        -offsets[0],
        0.0,
        1.0,
        -offsets[1],
        0.0,
        0.0
    ])

def random_flip_left_right(probability=0.5):
    """
    Flip an image left to right.

    Parameters
    ----------
    probability: float
        Probability to apply the flip.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_left_right(image),
                                  tf.image.flip_left_right(label)))
        return result
    return rand_flip

def random_flip_up_down(probability=0.5):
    """
    Flip an image vertically.

    Parameters
    ----------
    probability: float
        Probability to apply the flip.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_flip(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (tf.image.flip_up_down(image),
                                  tf.image.flip_up_down(label)))
        return result
    return rand_flip

def random_rotate(probability=0.5, max_angle=5.0):
    """
    Apply a random rotation.

    Parameters
    ----------
    probability: float
        Probability to apply a rotation.
    max_angle: float
        In radians. If applied, the image will be rotated by a random angle
        in the range [-max_angle, max_angle].

    Returns
    -------
    Augmentation function for the specified transform.
    """
    max_angle = max_angle * math.pi / 180.0
    def rand_rotation(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        theta = tf.random.uniform([], -max_angle, max_angle, tf.dtypes.float32)
        def apply_rotation():
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            transform = _rotation_transform(theta, height, width)
            return (
                _image_projective_transform(image, transform, fill_mode='REFLECT'),
                _image_projective_transform(label, transform, fill_mode='REFLECT')
            )
        result = tf.cond(r > probability, lambda: (image, label), apply_rotation)
        return result
    return rand_rotation

def random_translate(probability=0.5, max_pixels=7):
    """
    Apply a random translation.

    Parameters
    ----------
    probability: float
        Probability to apply the transform.
    max_pixels: int
        If applied, the image will be rotated by a random number of pixels
        in the range [-max_pixels, max_pixels] in both the x and y directions.

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_translate(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        t = tf.random.uniform([2], -max_pixels, max_pixels, tf.dtypes.float32)
        def apply_translate():
            transform = _translation_transform(t)
            return (
                _image_projective_transform(image, transform, fill_mode='REFLECT'),
                _image_projective_transform(label, transform, fill_mode='REFLECT')
            )
        result = tf.cond(r > probability, lambda: (image, label), apply_translate)
        return result
    return rand_translate

def random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5):
    """
    Apply a random brightness adjustment.

    Parameters
    ----------
    probability: float
        Probability to apply the transform.
    min_factor: float
    max_factor: float
        Brightness will be chosen uniformly at random from [min_factor, max_factor].

    Returns
    -------
    Augmentation function for the specified transform.
    """
    def rand_brightness(image, label):
        r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        t = tf.random.uniform([], min_factor, max_factor, tf.dtypes.float32)
        result = tf.cond(r > probability, lambda: (image, label),
                         lambda: (t * image,
                                  label))
        return result
    return rand_brightness

register_augmentation('random_flip_left_right', random_flip_left_right)
register_augmentation('random_flip_up_down', random_flip_up_down)
register_augmentation('random_rotate', random_rotate)
register_augmentation('random_translate', random_translate)
register_augmentation('random_brightness', random_brightness)
