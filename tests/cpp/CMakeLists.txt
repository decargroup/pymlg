include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_liegroups test_so3.cpp test_se3.cpp)
target_link_libraries(test_liegroups PRIVATE GTest::gtest_main Eigen3::Eigen)
target_include_directories(test_liegroups PRIVATE ${CMAKE_SOURCE_DIR}/pylie/cpp )
gtest_discover_tests(test_liegroups)