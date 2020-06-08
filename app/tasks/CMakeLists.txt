# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#####set(BIN_NAME LQR-Acrobot)

# The following 5 lines come from drake-shambhala
option(RUN_X11_TESTS "Run tests that require X11" OFF)
find_package(PythonInterp 2.7 EXACT MODULE REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${MUJOCO_PY_MJPRO_PATH}/cmake/modules")
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
set(BIN_NAME ilqr_control)
add_executable(${BIN_NAME} ${BIN_NAME}.cpp)
target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
#Standard libraries for GL
target_link_libraries(${BIN_NAME} glut)
target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
# Additional libraries from mujoco package
target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
target_link_libraries(${BIN_NAME} control_lib)
target_link_libraries(${BIN_NAME} util_lib)


set(BIN_NAME mppi_control_finger)
add_executable(${BIN_NAME} ${BIN_NAME}.cpp)
target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
#Standard libraries for GL
target_link_libraries(${BIN_NAME} glut)
target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
# Additional libraries from mujoco package
target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
target_link_libraries(${BIN_NAME} control_lib)
target_link_libraries(${BIN_NAME} util_lib)
target_link_libraries(${BIN_NAME} autodiff::autodiff)


set(BIN_NAME mppi_control_cartpole)
add_executable(${BIN_NAME} ${BIN_NAME}.cpp)
target_link_libraries(${BIN_NAME} ${LIB_MUJOCO})
#Standard libraries for GL
target_link_libraries(${BIN_NAME} glut)
target_link_libraries(${BIN_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
# Additional libraries from mujoco package
target_link_libraries(${BIN_NAME} libglew.so libglfw.so libglewegl.so libglewosmesa.so)
target_link_libraries(${BIN_NAME} control_lib)
target_link_libraries(${BIN_NAME} util_lib)
target_link_libraries(${BIN_NAME} autodiff::autodiff)
