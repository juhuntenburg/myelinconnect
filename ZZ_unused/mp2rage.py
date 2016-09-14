from nipype.pipeline.engine import Workflow, Node 
import nipype.interfaces.utility as util
from nipype.interfaces.mipav.developer import JistIntensityMp2rageMasking


def create_mp2rage_pipeline(name='mp2rage'):
    
    # workflow
    mp2rage = Workflow('mp2rage')
    
    # inputnode 
    inputnode=Node(util.IdentityInterface(fields=['inv2',
                                                  'uni',
                                                  't1map']),
               name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['uni_masked',
                                                   'background_mask',
                                                   ]),
                name='outputnode')
    
    # remove background noise
    background = Node(JistIntensityMp2rageMasking(outMasked=True,
                                            outMasked2=True,
                                            outSignal2=True), 
                      name='background')
    
    
    # connections
    mp2rage.connect([(inputnode, background, [('inv2', 'inSecond'),
                                              ('t1map', 'inQuantitative'),
                                              ('uni', 'inT1weighted')]),
                     (background, outputnode, [('outMasked2','uni_masked'),
                                               ('outSignal2','background_mask')]),
                     ])
    
    
    return mp2rage
