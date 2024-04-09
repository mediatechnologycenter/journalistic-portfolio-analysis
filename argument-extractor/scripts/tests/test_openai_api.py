# *  SPDX-License-Identifier: MIT
# *  © 2023-2024 ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Unit tests for OpenAI API
"""

import unittest
import openai

import os
from dotenv import load_dotenv, find_dotenv

class TestAuthentication(unittest.TestCase):
	def test_env_vars_exists(self):
		"""
		Test that OpenAI API keys are defined
		"""
		load_dotenv(find_dotenv())
		for var in ["OPENAI_API_KEY", "OPENAI_ORG_KEY"]:
			self.assertIsNotNone(
				os.getenv(var), 
				f"""
				OpenAI API keys are not defined! 
				Check either that .env file exists and has {var} variable defined,
				or make sure you export the variable in CLI
				"""
				)

	def test_env_vars_valid(self):
		"""
		Test that OpenAI API keys are valid
		"""
		load_dotenv(find_dotenv())
		openai.organization = os.getenv("OPENAI_ORG_KEY")
		openai.api_key = os.getenv("OPENAI_API_KEY")

		try:
			openai.Model.list()
		except openai.error.AuthenticationError:
			self.fail(
				"""
				OpenAI API keys are not valid! 
				Double-check that your keys match https://platform.openai.com/account/
				or generate new keys
				"""
				)

if __name__ == "__main__":
	unittest.main()