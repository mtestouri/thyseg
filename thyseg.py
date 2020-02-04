import logging
from cytomine import Cytomine

from cytomine.models import Project
from cytomine.models.image import ImageInstanceCollection
from cytomine.models.image import ImageInstance
from cytomine.models import AnnotationCollection, ImageInstanceCollection
 
host = "research.cytomine.be"
public_key = 'b8a99025-7dfa-41af-b317-eb98c3c55302'
private_key = 'd7c53597-0de4-4255-b7cd-9e3db60bddc2'
project_id = 77150529

with Cytomine(host=host, public_key=public_key, private_key=private_key) as conn:
    #print(conn.current_use)
    
    project = Project().fetch(id=project_id)
    #image = ImageInstance().fetch(77150617)
    #print(image.project)
    #exit()
    
    annotations = AnnotationCollection()
    annotations.project = project_id
    annotations.fetch()
    print(annotations)

    count = 0
    for annotation in annotations:
        #if(annotation.term is not None):
        if(annotation.image == 77150967):
            print("ID: {} | Image: {} | Term: {} | User: {}".format(
                annotation.id,
                annotation.image,
                annotation.term,
                annotation.user
            ))
            count += 1
    print(count)
