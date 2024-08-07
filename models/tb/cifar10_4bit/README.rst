cifar-10
========

Simple C++ reader for CIFAR-10 dataset

Usage
-----

You have to include cifar10_reader.hpp in your code:

.. code:: cpp

    #include "cifar/cifar10_reader.hpp"

And then, you can use the function :code:`read_dataset()` that returns a struct with a
vector of training images, one of test images, one of training labels and one of
test labels:

.. code:: cpp

   auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

The first two template arguments defines which container will be used for the
collections. The third argument is the type that is used to store a pixel (value
between 0 and 255) and the last one is the type that is used to store a label
(value between 0 and 9). The types in the example are the types by default, you
can use any STL container for the containers and any type that is convertible
from :code:`unsigned char` for the second.

Download the data
-----------------

The data files are not directly included in the data set. You must use the
download_cifar10.sh script in order to download them:

.. code:: bash

    ./download_cifar10.sh

License
-------

The header files are distributed under the terms of the MIT License. The
CIFAR-10 are not my property. If used in a paper, you'll need to cite the reference
paper, as indicated in the `official website <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
