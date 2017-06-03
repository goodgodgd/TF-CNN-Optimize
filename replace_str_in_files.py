import os
import shutil

def replace_checkpoint_path():
  root_dir = "/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints/my-fine-tuned"
  before_str = "/home/cideep/Work/tensorflow/checkpoints/my-fine-tuned"
  after_str = "/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints/my-fine-tuned"

  for root, dirs, files in os.walk(root_dir, topdown=False):
    for fname in files:
      if "checkpoint" in fname:
        replace_string(os.path.join(root, fname), before_str, after_str)


def replace_excuters():
  root_dir = "/home/cideep/Work/tensorflow/codes/excuters"
  before_str = "/home/cideep/Work/tensorflow/checkpoints/$"
  after_str = "/home/cideep/Work/tensorflow/checkpoints/Link-to-checkpoints/$"

  for root, dirs, files in os.walk(root_dir, topdown=False):
    for fname in files:
      if fname.endswith(".sh"):
        replace_string(os.path.join(root, fname), before_str, after_str)

  before_str = "/home/cideep/Work/tensorflow/datasets/$"
  after_str = "/home/cideep/Work/tensorflow/datasets/Link-to-datasets/$"

  for root, dirs, files in os.walk(root_dir, topdown=False):
    for fname in files:
      if fname.endswith(".sh"):
        replace_string(os.path.join(root, fname), before_str, after_str)


def replace_string(filename, befstr, aftstr):
  print("file to change", filename)
  bakfilename = filename+'.bak'
  shutil.copy2(filename, bakfilename)

  f = open(filename, 'r')
  filedata = f.read()
  f.close()

  newdata = filedata.replace(befstr, aftstr)

  f = open(filename, 'w')
  f.write(newdata)
  f.close()


if __name__ == '__main__':
  # replace_checkpoint_path()
  replace_excuters()
