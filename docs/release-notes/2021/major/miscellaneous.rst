Miscellaneous
^^^^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without the
   a space between the colon and number!

CMake multichoice and trivalue variables are now case sensitive
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The multichoice variables for settings such as GMX_FFT_LIBRARY,
GMX_SIMD, GMX_USE_RDTSCP and GMX_USE_LMFIT are now case-sensitive,
and all trivalue variables now use "Auto" for the third choice
(GMX_HWLOC, GMX_BUILD_HELP, and GMX_LOAD_PLUGINS). CMake has
always had case-sensitive string handling, but previously we tried
to hide this. Unfortunately this solution was fragile internally,
so if you get an error message about invalid choice, please adjust
the case to match one of the listed available choices.
