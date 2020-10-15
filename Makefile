all:
	yarn prepare

pub:
	yarn version --minor
	npm publish
