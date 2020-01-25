run-job:
	# Run a kubernetes job with our image, prefix with USERNAME
	envsubst < job.yml | kubectl create -f -

delete-all:
	# Delete all jobs
	kubectl delete jobs --all

get-pods:
	# Get all pods
	kubectl get pods

get-jobs:
	# Get all jobs
	kubectl get jobs

kube-bash:
	# Runs bash within the pod: pod name always includes $USER
	# WARNING: one POD needs to be running
	# TODO: add support for specifying POD ID
	POD=$$(kubectl get pods -o name --no-headers=true); \
	echo pod found:$$POD; \
	kubectl exec -it $$POD -- /bin/bash
kube-log:
	# logs for pod
	POD=$$(kubectl get pods -o name --no-headers=true); \
	kubectl logs -f $$POD
docker-make:
	# Make docker image and push
	sudo docker build -t ${img} . 
	IMGID=$$(sudo docker images --filter=reference=${img} --format "{{.ID}}"); \
	sudo docker tag $$IMGID ${DOCKERHUB_USERNAME}/${img}:${version}; \
	sudo docker push ${DOCKERHUB_USERNAME}/${img}:${version}
monitor:
	# Run nvidia monitor in a loop to monitor GPU usage
	kubectl exec -it $(USER)-pod -- nvidia-smi --loop=5

