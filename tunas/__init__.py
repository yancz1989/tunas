# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-04-12 10:06:08
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-05 11:43:41

from . import core
from . import arch
from . import layers
from . import util

# This file deals with config file, loading and reconfig for tunas library.
# The config file and  temporary file locates at '/home/your_name/.tunas' as
# default. The entire lib use float32 as  floatX, and epsilon as 1e-7. You may
# change as you need when install from source. We will use tensorflow on the
# first atempt for its more advanced support for distributed and  multi-card.
# If tensorflow is not installed, Theano is tried as an alternative. We
# require users to have either tensorflow or Theano installed and cudnn must
# be installed for gpu usage.

##############
# Code Style #
##############
# In this project, we will use _UPPERCASE_ as global variable, _lowercase_ as
# local variable which not accessable for users. Common variable are written
# in lowercase and separate with '_'.  Functions is named the same way with
# common variable whereas classes are named in non-underline separated words
# with leading character uppercase like 'ClassDef'. Comment should be insert
# at  the line just after the defination of function and class. Certain
# complicated routain should be commented with proper explain. This is
# documented for readers of this project. In comment, parameter should begin
# with '@' to distinguished with common words.



