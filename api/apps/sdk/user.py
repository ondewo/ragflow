#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from api.utils.api_utils import token_required


@manager.route("/models", methods=["POST"])  # noqa: F821
@token_required
async def add_model(tenant_id: str):
	"""
	Add a new model or model factory.

	Allows adding models in two different ways:
	- individual model: add a single custom (local/self-hosted model)
	- model factory: add all pre-configured models of a model factory (from model service providers like OpenAI, Anthropic, ...)

	General request body parameters:
	- model_factory (str, required): Model factory name.
	- api_key (str): API key to use for the added model(s).
	- base_url (str): Base URL of the model(s). If not provided, factory default will be used if available.

	Individual model request body parameters:
	- model_name (str, required):

	Model factory request body parameters:

	Response body:
		Empty on success.
	"""
	...


@manager.route("/models", method=["GET"])  # noqa: F821
@token_required
async def list_models(tenant_id: str):
	...


@manager.route("/models", method=["DELETE"])  # noqa: F821
@token_required
async def remove_models(tenant_id: str):
	...


@manager.route("/models/defaults", method=["POST"])  # noqa: F821
@token_required
async def set_default_models(tenant_id: str):
	...


@manager.route("/models/defaults", method=["GET"])  # noqa: F821
@token_required
async def get_default_models(tenant_id: str):
	...
