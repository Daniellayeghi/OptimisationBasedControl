
#ifndef OPTCONTROL_MUJOCO_SENSORS_H
#define OPTCONTROL_MUJOCO_SENSORS_H
#include "mujoco.h"

class Sensors
{
public:
    Sensors(const mjModel* m, mjData *d);
    static void set_instance(Sensors *sensor);
    static void callback_wrapper(const mjModel* m, mjData* d, int);

private:
    void wrap_to_PI();
    const mjModel* _m;
    mjData* _d;
};


#endif //OPTCONTROL_MUJOCO_SENSORS_H
