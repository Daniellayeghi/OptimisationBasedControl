SET(UI_HEADER_LIST uitools.h)
SET(UI_SOURCE_LIST uitools.c)
add_library(ui_lib ${UI_SOURCE_LIST} ${UI_HEADER_LIST})
