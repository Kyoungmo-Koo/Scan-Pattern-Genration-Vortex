Vortex_install.py - Code used for installing Vortex.

SPGV_test.py - Try many scan pattern generation functions of Vortex. 
               Various methods of generating RasterScan, RadialScan and FreeFormScan patterns.
               
SPGV_5000.py - Same as SPGV_test, but only making scan patterns with the length of 5000.

SPGV_Vortex2Hexa.py - Change scan patterns that are made by vortex into hexadecimal txt files 
                      that can be used for digital interface.
                      
Scan_Pattern_Amplifier - Amplify the scan pattern in domain of servo driver.

oct.py, engine.py -> Consider when generating scan pattern for image generation. Vortex is scanning image based on
                      RasterScan + FreeFormScan, RadialScan + FreeFormScan ways.
