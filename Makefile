include .env

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* icangetyoursmile/*.py

black:
	@black scripts/* icangetyoursmile/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr icangetyoursmile-*.dist-info
	@rm -fr icangetyoursmile.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
# TRAINER : LOCAL AND GOOGLE IA FUNCTION
# ----------------------------------
#
# These environment variables are define in .env
# BUCKET_NAME = name of bucket on GCP
# BUCKET_PACKAGE_FOLDER = name of GCP folder containing the package
# BUCKET_STORAGE_FOLDER = name of GCP folder where training results and data are stored
# PACKAGE_NAME = computer path to the folder package containing the file to run
# FILENAME = name of file to run
# PYTHON_VERSION = python version
# RUNTIME_VERSION = libraries version
# REGION = physical region of the server on which to train

run_locally:
  @python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
  gcloud ai-platform jobs submit training ${JOB_NAME} \
    --job-dir gs://${BUCKET_NAME}/${BUCKET_PACKAGE_FOLDER} \
    --package-path ${PACKAGE_NAME} \
    --module-name ${PACKAGE_NAME}.${FILENAME} \
    --python-version=${PYTHON_VERSION} \
    --runtime-version=${RUNTIME_VERSION} \
    --region ${REGION} \
    --stream-logs

clean:
  @rm -f */version.txt
  @rm -f .coverage
  @rm -fr */__pycache__ __pycache__
  @rm -fr build dist *.dist-info *.egg-info
  @rm -fr */*.pyc


set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

FRAMEWORK=scikit-learn


##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload

