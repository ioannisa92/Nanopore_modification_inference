run-job:
	# Run a kubernetes job with our image, prefix with USERNAME
	envsubst < job.yml | kubectl create -f -
delete-pod:
	# Delete specific pod
	kubectl delete pod ${pod}
delete-alljobs:
	# Delete all jobs
	kubectl delete jobs --all
delete-job:
	# Delete specific pod
	kubectl delete jobs ${job}
get-pods:
	# Get all pods
	kubectl get pods

get-jobs:
	# Get all jobs
	kubectl get jobs

kube-bash:
	# Runs bash within specified pod
	kubectl exec -it ${pod} -- /bin/bash
kube-log:
	# logs for pod
	
	kubectl logs -f ${pod}
docker-make:
	# Make docker image and push
	sudo docker build -t ${img} . 
	IMGID=$$(sudo docker images --filter=reference=${img} --format "{{.ID}}"); \
	sudo docker tag $$IMGID ${DOCKERHUB_USERNAME}/${img}:${version}; \
	sudo docker push ${DOCKERHUB_USERNAME}/${img}:${version}
monitor:
	# Run nvidia monitor in a loop to monitor GPU usage
	kubectl exec -it ${pod} -- nvidia-smi --loop=1
docker-delete-none:
	sudo docker images | grep none | awk '{ print $3; }' | xargs docker rmi

cp-run:
	# Copies the script and runs in on the pod-maybe?
	# DOES NOT WORK
	envsubst < pod.yml | kubectl create -f - ; \
	kubectl wait --for=condition=running pod/${pod} --timeout=120s; \
	kubectl cp ${script} stuartlab/${pod}:/root && \
		kubectl exec -it ${pod} -- python ${script}
    
