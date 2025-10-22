cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

PROJECT(muography
  LANGUAGES CXX
  )

find_package( DD4hep REQUIRED COMPONENTS DDCore DDG4 )
#find_package(Acts REQUIRED COMPONENTS Core PluginIdentification PluginTGeo PluginDD4hep )


#-----------------------------------------------------------------------------------
#set(lib_name zdc_prototype)
dd4hep_add_plugin(${PROJECT_NAME} SOURCES src/*.cpp)

target_link_libraries(${PROJECT_NAME}
  PUBLIC DD4hep::DDCore  DD4hep::DDRec
)
install(DIRECTORY compact
    DESTINATION share/${PROJECT_NAME}
    )
install(DIRECTORY src
    DESTINATION share/${PROJECT_NAME}
    )

#-----------------------------------------------------------------------------------
file(GLOB MUOGRAPHY "*.xml")
INSTALL(FILES ${MUOGRAPHY}
	DESTINATION share/${PROJECT_NAME}
	)
