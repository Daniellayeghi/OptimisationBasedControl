#include <array>
#include <random>
#include "mujoco.h"
#include "mujoco_utils.h"

void MujocoUtils::populate_obstacles(const int start_id,
                                     const int end_id,
                                     const std::array<double, 6> &bounds,
                                     const mjModel *model){
    const constexpr int geo_dims = 3;
    using namespace std;

    array<double, geo_dims> random_pos{};
    auto random_iid_array = [](array<double, geo_dims> &result, const array<double, geo_dims * 2> &bounds) {
        random_device r;
        default_random_engine generator(r());
        for (auto dim = 0; dim < geo_dims; ++dim) {
            uniform_real_distribution<double> distribution(bounds[dim * 2], bounds[dim * 2 + 1]);
            result[dim] = distribution(generator);
        }
    };

    // Assumes that all obstacles are contiguous
    auto total_obs = end_id - start_id + 1;
    auto num_geoms = static_cast<int>(total_obs / geo_dims);
    auto bodies = 0;
    for (auto id = 0; id < num_geoms + 1; id++) {
        bodies = id * geo_dims;
        random_iid_array(random_pos, bounds);
        model->body_pos[start_id + bodies] = random_pos[0];
        model->body_pos[start_id + bodies + 1] = random_pos[1];
        model->body_pos[start_id + bodies + 2] = random_pos[2];
    }
}