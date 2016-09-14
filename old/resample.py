# imports for pipeline
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio

with open('/scr/ilz3/7tresting/surfaces/surf_rh/rh.txt', 'r') as f:
    surfaces = [line.strip() for line in f]
    
# local base and output directory (should usually not change)
data_dir = '/scr/ilz3/7tresting/surfaces/surf_rh/'
base_dir = '/scr/ilz3/7tresting/working_dir/'
out_dir = '/scr/ilz3/7tresting/surfaces/surf_rh/'

# workflow
resamp = Workflow(name='resamp')
resamp.base_dir = base_dir
resamp.config['execution']['crashdump_dir'] = resamp.base_dir + "/crash_files"


infosource=Node(util.IdentityInterface(fields=['surf']),
                name='inforsource')
infosource.iterables=[('surf', surfaces)]

# select files
templates={'surface': '{surf}'}
selectfiles = Node(nio.SelectFiles(templates,
                                   base_directory=data_dir),
                   name="selectfiles")
resamp.connect([(infosource, selectfiles, [('surf', 'surf')])])

# resample to 0.8 mm isotropic
iso=Node(fsl.FLIRT(apply_isoxfm=0.8,
                   interp='spline'),
         name='iso_sampling')

resamp.connect([(selectfiles, iso, [('surface', 'in_file'),
                                    ('surface', 'reference')])])

# sink
sink = Node(nio.DataSink(base_directory=out_dir,
                         parameterization=False),
                name='sink')

resamp.connect([(iso, sink, [('out_file', 'resample.@resamp')])
                ])

resamp.run()#plugin='CondorDAGMan')