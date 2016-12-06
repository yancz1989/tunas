# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-12-04 11:52:23
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-12-04 15:43:27

import argparse
import json
import os

def init(args):
  # write default configuration to config.json
  config = {'floatx' : args.floatx, 'eps' : args.eps, 'arch' : args.arch, 'device' : args.device}
  fconfig = os.path.expanduser('~') + '/.tunas'
  with open(fconfig, 'w') as f:
    f.write(json.dumps(config) + '\n')
    f.close()

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--floatx', '-F', help = 'Definition of floatX, valid options are float32 and float64.', default = 'float32', choices = ['float32', 'float64'])
  parser.add_argument('--eps', '-E', help = 'Assign an eps precision, in float, i.e., 1e-6.', default = float('1e-6'), choices = ['1e-4', '1e-5', '1e-6', '1e-7'])
  parser.add_argument('--arch', '-A', help = 'Choice of architecture, current options include tensorflow and theano.', default = 'tensorflow', choices = ['tensorflow', 'theano'])
  parser.add_argument('--device', '-D', help = 'Default device, set to GPU No. if you want to use specified GPU device, and empty if only cpu is availiable. Multiple GPUs are separated with comma. i.e. \'0,1,2\'\n WARNING: if no devices is declared, for tensorflow, all GPU will be occupied.', default = '')
  init(parser.parse_args())


if __name__ == '__main__':
  parse()