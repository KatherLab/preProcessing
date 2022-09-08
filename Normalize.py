"""
@author: Narmin Ghaffari Laleh <narminghaffari23@gmail.com> - Nov 2020

Quick fix to run on Linux again introduced September 8th 2022 by Omar El Nahhas (omarelnahhas1337@gmail.com)
For the spams package, I recommend running this in a conda virtual environment and install everything
Spams cannot be installed with pypi successfully (08/09/2022)

"""
##############################################################################
from multiprocessing.dummy import Pool as ThreadPool
from operator import itemgetter
import sys
import stainNorm_Macenko
import multiprocessing
import os
import cv2
import numpy  as np
from pathlib import Path
from itertools import repeat
import argparse
#TODO: create process bar that works in multiprocessing env
#from tqdm import tqdm

# global inputPath
# global outputPath
#global normalizer # remove normalizer as global variable

##############################################################################



def Normalize_Main(inputPath, outputPath, item, normalizer): 

    outputPathRoot = os.path.join(outputPath, item)
    inputPathRoot = os.path.join(inputPath, item)
    inputPathRootContent = os.listdir(inputPathRoot)
    if not len(inputPathRootContent) == 0:
        if not os.path.exists(outputPathRoot):
            os.mkdir(outputPathRoot)
            temp = os.path.join(inputPath, item)
            tempContent = os.listdir(temp)
            tempContent = [i for i in tempContent if i.endswith('.jpg')]
            for tempItem in tempContent:
                img = cv2.imread(os.path.join(inputPathRoot, tempItem))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                edge  = cv2.Canny(img, 40, 100)
                edge = edge / np.max(edge) if np.max(edge) != 0 else 0
                edge = (np.sum(np.sum(edge)) / (img.shape[0] *img.shape[1])) * 100 if np.max(edge) != 0 else 0
                #print(edge)
                if edge > 2:
                    try:
                        nor_img = normalizer.transform(img)
                        cv2.imwrite(os.path.join(outputPathRoot, tempItem), cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))
                    except:
                        print('Failed to normalize the tile {}.'.format(tempItem))
                    
##############################################################################
                        
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

###############################################################################
    
def Normalization(inputPath: Path, outputPath: Path, sampleImagePath: Path, num_threads: int) -> None:
    
    inputPathContent = os.listdir(inputPath)
    normPathContent = os.listdir(outputPath)
    
    remainlList = []
    for i in inputPathContent:
        if not i in normPathContent:
            remainlList.append(i)
            
    inputPathContent = [i for i in remainlList if not i.endswith('.bat')]
    inputPathContent = [i for i in inputPathContent if not i.endswith('.txt')]
    
    target = cv2.imread(sampleImagePath)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
    pool = ThreadPool(num_threads)
    normalizer = stainNorm_Macenko.Normalizer()
    normalizer.fit(target)
    # old function, only passed inputpath
    # pool.map(Normalize_Main, inputPathContent)
    
    #quick fix uses starmap, which passes the arguments as iterative objects.
    #repeat is used for constants

    pool.starmap(Normalize_Main, zip((repeat(inputPath)), repeat(outputPath), inputPathContent, repeat(normalizer)))
    pool.close()
    pool.join()

if __name__ == '__main__':
    #parsing all arguments from the command line
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-ip", "--inputPath", help="Input path of the to-be-normalised tiles", type=Path, required=True)
    requiredNamed.add_argument("-op", "--outputPath", help="Output path to store normalised tiles", type=Path, required=True)
    parser.add_argument("-si", "--sampleImagePath", help="Image used to determine the colour distribution, uses GitHub one by default", type=Path)
    parser.add_argument("-nt", "--threads", help="Number of threads used for processing, 2 by default", type=int)
    args = parser.parse_args()

    #calling the Normalization function with defined parameters
    Normalization(  args.inputPath,
                    args.outputPath, 
                    args.sampleImagePath if args.sampleImagePath != None else 'normalization_template.jpg', 
                    args.threads if args.threads != None else 2)
