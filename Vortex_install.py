
#fter CUDA 12.1.0 Install 

pip install vortex-oct-cuda12x --index-url https://vortex-oct.dev/develop

pip install vortex-oct-cuda12x vortex-oct-tools -f https://www.vortex-oct.dev/

import vortex
print('OK', vortex.__version__, vortex.__feature__)

####################################################Setup###########################################
