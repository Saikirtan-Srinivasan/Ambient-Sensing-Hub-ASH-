/***************************************************************************//**
 * SI917 + MLX90640 + TFLite Micro People Counting
 ******************************************************************************/

#include "sl_sleeptimer.h"
#include "app_assert.h"
#include "sparkfun_mlx90640.h"
#include "sparkfun_mlx90640_config.h"

#if (defined(SLI_SI917))
#include "sl_i2c_instances.h"
#include "rsi_debug.h"

#define app_printf(...) DEBUGOUT(__VA_ARGS__)
#define I2C_INSTANCE_USED SL_I2C2
static sl_i2c_instance_t i2c_instance = I2C_INSTANCE_USED;
static mikroe_i2c_handle_t app_i2c_instance = &i2c_instance;
#endif

/*********** TFLite Includes ***********/
#include "sl_tflite_micro_init.h"
#include "tensorflow/lite/c/common.h"

#include "people_counter_int8_generated.h"

/*********** Configuration ***********/
#define READING_INTERVAL_MSEC 250
#define TENSOR_ARENA_SIZE (50 * 1024)

/*********** Thermal Frame ***********/
static float mlx90640_image[SPARKFUN_MLX90640_NUM_OF_PIXELS];

/*********** Timer ***********/
static sl_sleeptimer_timer_handle_t app_timer_handle;
static volatile bool app_timer_expire = false;

/*********** Tensor Arena ***********/
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
extern const unsigned char people_counter_int8_generated[];
/*********** TFLite Globals ***********/
static TfLiteTensor* input;
static TfLiteTensor* output;

/*********** Stability Filter ***********/
static int last_people_count = -1;
static int stable_counter = 0;

static void app_timer_cb(sl_sleeptimer_timer_handle_t *handle, void *data);
static tflite::MicroErrorReporter micro_error_reporter;


/***************************************************************************//**
 * Initialize Application
 ******************************************************************************/
void app_init(void)
{
  sl_status_t sc;

  /******** MLX90640 INIT ********/
  sc = sparkfun_mlx90640_init(app_i2c_instance,
                              SPARKFUN_MLX90640_I2C_ADDRESS);

  if (sc != SL_STATUS_OK) {
    app_printf("MLX90640 init failed!\n");
    return;
  }

  sparkfun_mlx90640_set_refresh_rate(0x03);
  app_printf("Thermal Camera Ready\n");

  /******** TFLite Setup ********/



  // Initialize TFLite Micro engine
   sl_tflite_micro_init();

   app_printf("TFLite Ready\n");
  /******** Start Periodic Timer ********/
  sl_sleeptimer_start_periodic_timer_ms(
      &app_timer_handle,
      READING_INTERVAL_MSEC,
      app_timer_cb,
      NULL,
      0,
      0);

  app_printf("People Counting Started\n");
}

/***************************************************************************//**
 * Main Loop
 ******************************************************************************/
void app_process_action(void)
{
  if (!app_timer_expire)
    return;

  app_timer_expire = false;

  /******** Capture Thermal Frame ********/
  sparkfun_mlx90640_get_image_array(mlx90640_image);

  /******** Get TFLite Tensors ********/
  TfLiteTensor* input = sl_tflite_micro_get_input_tensor();
  TfLiteTensor* output = sl_tflite_micro_get_output_tensor();


  if (input == NULL || output == NULL) {
    app_printf("TFLite not initialized!\n");
    return;
  }

  /******** Preprocess + Quantize ********/
  float min_temp = 20.0f;
  float max_temp = 40.0f;

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  for (int i = 0; i < 768; i++) {

    float normalized =
        (mlx90640_image[i] - min_temp) / (max_temp - min_temp);

    if (normalized < 0) normalized = 0;
    if (normalized > 1) normalized = 1;

    input->data.int8[i] =
        (int8_t)(normalized / input_scale + input_zero_point);
  }

  /******** Inference ********/
  if (sl_tflite_micro_get_interpreter()->Invoke() != SL_STATUS_OK) {
    app_printf("Inference failed!\n");
    return;
  }

  /******** Get People Count ********/
  int num_classes = output->dims->data[1];

  int8_t max_score = -128;
  int people_count = 0;

  for (int i = 0; i < num_classes; i++) {
    int8_t score = output->data.int8[i];

    if (score > max_score) {
      max_score = score;
      people_count = i;
    }
  }

  /******** Stability Filter ********/
  if (people_count == last_people_count) {
    stable_counter++;
  } else {
    stable_counter = 0;
  }

  if (stable_counter >= 2) {
    app_printf("People Count: %d\n", people_count);
  }

  last_people_count = people_count;
}

/***************************************************************************//**
 * Timer Callback
 ******************************************************************************/
static void app_timer_cb(sl_sleeptimer_timer_handle_t *handle, void *data)
{
  (void)handle;
  (void)data;
  app_timer_expire = true;
}
