# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-05-05 21:18:05
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-06-13 09:37:25

# This file implement interfaces of basic operating system operations, and
# support for storage using hdf5.
def dict2hdf5(dic, fn):
  def subdict2hdf5(h5file, path, dic):
    for key, item in dic.items():
      if isinstance(item, (np.ndarray, np.int64, np.int8, np.int32,
        np.float32, np.float64, str, bytes, int, float, list)):
        h5file[path + str(key)] = item
      elif isinstance(item, dict):
        subdict2hdf5(h5file, path + str(key) + '/', item)
      else:
        raise ValueError('Cannot save %s type'%type(item))

  with h5py.File(fn, 'w') as h5file:
    subdict2hdf5(h5file, '/', dic)

def hdf52dict(fn):
  def hdf52dictsub(h5file, path):
    ret = {}
    for key, item in h5file[path].items():
      if isinstance(item, h5py._hl.dataset.Dataset):
        ret[str(key)] = item.value
      elif isinstance(item, h5py._hl.group.Group):
        ret[str(key)] = hdf52dictsub(h5file, path + str(key) + '/')
    return ret

  with h5py.File(fn, 'r') as h5file:
    return hdf52dictsub(h5file, '/')


def list2hdf5(lst, fn):
  dic = {}
  for i in range(len(lst)):
    dic[str(i)] = lst[i]
  dict2hdf5(dic, fn)

def hdf52list(fn):
  dat = []
  dic = hdf52dict(fn)
  for i in range(len(dic)):
    dat.append(dic[str(i)])
  return dat

def generator2hdf5(fn, g, dname, size, dtype = _FLOATX_):
  '''
    @fn, filename of h5 file.
    @g, generator need to be stored.
    @dname, string, dataset name.
    @size, list, size of entire data.
    @dtype, data type, default as _FLOATX_
  '''
  with h5py.File(fn, 'w') as f:
    dat = f.create_dataset(dn, sz, dtype = dtype)
    for i in range(size[0]):
      dat[i, :] = g.next()
