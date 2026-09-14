FROM tensorflow/serving:latest

# Salin model hasil Pusher ke image
COPY ./serving_model/johnrahmadi-pipeline /models/johnrahmadi-pipeline

# Variabel environment untuk TF Serving
ENV MODEL_NAME=johnrahmadi-pipeline
ENV MODEL_BASE_PATH=/models

# Railway/Heroku menginject PORT secara dinamis, jadi start.sh dipakai untuk binding port
COPY ./monitoring/prometheus.config /model_config/prometheus.config

RUN echo '#!/bin/bash \n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT:-8501} \
--model_name=${MODEL_NAME} \
--model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=/model_config/prometheus.config \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
