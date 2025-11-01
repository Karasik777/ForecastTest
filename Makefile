.PHONY: docker-build docker-shell docker-fetch docker-process docker-predict docker-evaluate docker-consensus docker-pipeline plots consensus-plot

docker-build:
	docker compose build

docker-shell:
	docker compose run --rm shell

docker-fetch:
	docker compose run --rm stage_fetch --env SKIP_VENV=1 $(ARGS)

docker-process:
	docker compose run --rm stage_process --env SKIP_VENV=1 $(ARGS)

docker-predict:
	docker compose run --rm stage_predict --env SKIP_VENV=1 $(ARGS)

docker-evaluate:
	docker compose run --rm stage_evaluate --env SKIP_VENV=1 $(ARGS)

docker-consensus:
	docker compose run --rm stage_consensus --env SKIP_VENV=1 $(ARGS)

docker-pipeline:
	./docker/pipeline.sh $(ARGS)

plots:
	./exec.sh --plots $(ARGS)

consensus-plot:
	./exec.sh --consensus-plot $(ARGS)
