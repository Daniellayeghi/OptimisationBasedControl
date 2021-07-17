#include "sensors.h"

static Sensors *static_sensor;


Sensors::Sensors(const mjModel *m, mjData *d) : _m(m), _d(d)
{

}


void Sensors::set_instance(Sensors *sensor)
{
    static_sensor = sensor;
}


void Sensors::callback_wrapper(const mjModel* m, mjData* d, int)
{
    static_sensor->wrap_to_PI();
}


void Sensors::wrap_to_PI()
{

}

