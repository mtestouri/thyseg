# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from shapely.geometry import shape, box, Polygon, MultiPolygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

from cytomine import Cytomine
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection

from PIL import Image

import argparse
import json


#Code is quite messy but working if you download a Stardist model locally (demo model could also be used).
#For illustration purposes.
#Example on DEMO-SEGMENTATION-TISSUE project with jsnow user (need to provide credentials.json)
#python cytomine-stardist.py -p 528050 -i 528132 -r 154121648 -t 154122471 -u 263676 -wd "/tmp/cytomine/"

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
                        description="Simple example to add Stardist (https://github.com/mpicbg-csbd/stardist) nuclei detections within a Cytomine ROI to an image.")
    parser.add_argument('-p', type=int, required=True,
                        help="the project id")
    parser.add_argument('-i', type=int, required=True,
                        help="the image id")
    parser.add_argument('-s', type=int, required=False,
                        help="the slice id")
    parser.add_argument('-u', type=int, required=True,
                        help="the user id layer")
    parser.add_argument('-r', type=int, required=True,
                        help="the id_term of ROI annotation(s) where to detect nuclei")
    parser.add_argument('-t', type=int, required=True,
                        help="the nuclei term id")
    parser.add_argument('-wd', type=str, required=True,
                        help="the local working directory where ROI images are downloaded")
    
    args = parser.parse_args()


    #Loading pre-trained Stardist model
    np.random.seed(6)
    lbl_cmap = random_label_cmap()
    #Stardist H&E model downloaded from https://github.com/mpicbg-csbd/stardist/issues/46
    #Stardist H&E model downloaded from https://drive.switch.ch/index.php/s/LTYaIud7w6lCyuI
    model = StarDist2D(None, name='2D_versatile_HE', basedir='models')   #use local model file in ~/models/2D_versatile_HE/

    #Read local Cytomine credentials file with public/private key and host server URL (https)
    cred = json.load(open('credentials.json'))
    #cred = json.load(open('credentials-jsnow.json'))

    #Open connection to Cytomine server
    with Cytomine(host=cred['host'], 
                  public_key=cred['public_key'], 
                  private_key=cred['private_key'], verbose=0) as conn:


        #Get list of images in Cytomine project
        image_instances = ImageInstanceCollection().fetch_with_filter("project", args.p)
        for image in image_instances:
            if image.id==args.i:
                print("------------- Image ID: {} | Width: {} | Height: {} | Resolution: {} | Magnification: {} | Filename: {}".format(
                    image.id, image.width, image.height, image.resolution, image.magnification, image.filename
                ))
                cytomine_image=image


        #Clean/delete existing annotations in image, by user, with term (for testing, use own annotation layer, so cleaning at each run)
        del_annotations = AnnotationCollection()
        del_annotations.image = args.i
        del_annotations.user = args.u
        del_annotations.project = args.p
        del_annotations.term = args.t
        del_annotations.fetch()
        print("-------Deleting old annotations---------------")
        print(del_annotations)
        for annotation in del_annotations:
            annotation.delete()


        #Dump ROI annotations from Cytomine server to local images
        roi_annotations = AnnotationCollection()
        roi_annotations.project = args.p
        roi_annotations.term = args.r
        roi_annotations.image = args.i
        roi_annotations.showWKT = True
        roi_annotations.fetch()
        print(roi_annotations)
        for roi in roi_annotations:
            #Get Cytomine ROI coordinates for remapping to whole-slide
            print("----------------------------ROI------------------------------")
            roi_geometry = wkt.loads(roi.location)
            print("ROI Geometry from Shapely: {}".format(roi_geometry))
            print("ROI Bounds")
            print(roi_geometry.bounds)
            minx=roi_geometry.bounds[0]
            miny=roi_geometry.bounds[3]
            print(minx,miny)
            #Dump ROI image in PNG
            roi.dump(dest_pattern=os.path.join(args.wd,"{project}","{id}","{id}.png"), mask=True, alpha=True)
            
            
            #Stardist works with TIFF images without alpha channel, conversion
            working_path=os.path.join(args.wd,str(args.p)+'/'+str(roi.id)) #'/home/.../{project}/{image}/'
            print(working_path)
            im=Image.open(os.path.join(working_path+'/'+str(roi.id)+'.png'))
            bg = Image.new("RGB", im.size, (255,255,255))
            #bg.paste(im,im) #if we want to use rectangular roi without taking into account alpha mask
            #Flattening PNG alpha mask to TIFF RGB
            bg.paste(im,mask=im.split()[3])
            roi_filename=os.path.join(working_path+'/'+str(roi.id)+'.tif')
            bg.save(roi_filename,quality=100)

            X_files = sorted(glob(working_path+'/'+str(roi.id)+'*.tif'))
            X = list(map(imread,X_files))
            print("X image dimension: %d, number of images: %d" %(X[0].ndim,len(X)))

            n_channel = 3 if X[0].ndim == 3 else X[0].shape[-1]
            axis_norm = (0,1)   # normalize channels independently
            #axis_norm = (0,1,2) # normalize channels jointly
            if n_channel > 1:
                print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

            #Going over images in directory (in our case: one ROI per directory)
            for x in range(0,len(X)):
                print("------------------- File %d: %s" %(x,roi_filename))
                img = normalize(X[x], 1,99.8, axis=axis_norm)
                labels, details = model.predict_instances(img)
                #Save label image locally for inspection
                matplotlib.image.imsave(working_path+'/'+str(roi.id)+os.path.basename(roi_filename)+"_label_"+str(len(details['coord']))+".png", labels)
                print("Number of detected polygons: %d" %len(details['coord']))
                cytomine_annotations = AnnotationCollection()
                for pos,polygroup in enumerate(details['coord'],start=1):
                    #print("polygroup %d len: %d %s" %(pos,len(polygroup),polygroup))
                    #print("polygroup %d len: %d" %(pos,len(polygroup)))
                    #polygroup is a pair of arrays, y[] and x[] coordinates
                    #Converting to Shapely annotation
                    points = list()
                    for i in range(len(polygroup[0])):
                        #print("minx: %d" %minx)
                        #print("polygroupx: %d" %polygroup[1][i])
                        #print("cytomine_image.height: %d" %cytomine_image.height)
                        #print("miny: %d" %miny)
                        #print("polygroupy: %d" %polygroup[0][i])
                        #print("ROI Bounds")
                        #print(roi_geometry.bounds)
                        #Cytomine cartesian coordinate systems, (0,0) is bottom left corner
                        #Mapping polygon detection coordinates to ROI in whole slide images
                        p = Point(minx+polygroup[1][i],miny-polygroup[0][i])
                        points.append(p)

                    annotation = Polygon(points)
                    #print(annotation)
                    #Send annotation to Cytomine server
                    #Add annotation one by one (one http request per annotation), with term. Slowish.
                    #print(annotation.is_valid)
                    cytomine_annotation = Annotation(location=annotation.wkt, id_image=args.i, id_project=args.p).save()
                    #Add term to annotation to Cytomine server
                    AnnotationTerm(cytomine_annotation.id, args.t).save()

                    #Alternative: Append to Annotation collection 
                    #cytomine_annotations.append(Annotation(location=annotation.wkt, id_image=args.i, id_project=args.p))
                    
                #Alternative (faster): add collection of annotations (without term) in one http request
                #cytomine_annotations.save()

