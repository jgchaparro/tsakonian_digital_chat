import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APITimeoutError
from streamlit.testing.v1 import AppTest

from pipeline import (DEFAULT_NIM_MODEL, DEFAULT_OLLAMA_MODEL, KRIKRI_MODEL, GREEK_SYSTEM,
                      TRANSLATION_SYSTEM, ask_nim, ask_ollama, translate)


def ollama_response(text="translation", **extra):
    return httpx.Response(
        200, json={"response": text, **extra},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )


def nim_response(text="Greek reply", reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=reason,
    )])


def krikri_response(text="Απάντηση", **extra):
    return httpx.Response(
        200, json={"message": {"content": text}, **extra},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Keep app tests independent of this computer's provider/model choices.
        settings_patch = patch("dotenv.dotenv_values", return_value={
            "GREEK_BACKEND": "nim", "NIM_MODEL": DEFAULT_NIM_MODEL,
            "OLLAMA_CHAT_MODEL": DEFAULT_OLLAMA_MODEL,
        })
        self.settings = settings_patch.start()
        self.addCleanup(settings_patch.stop)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_greek_text_is_shown_below_messages_during_failure_and_after_retry(self, client_class, post):
        create = client_class.return_value.chat.completions.create
        create.side_effect = httpx.ConnectError("Reply service unavailable")
        post.return_value = ollama_response("Ελληνική μετάφραση <script>unsafe</script>")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.chat_input[0].set_value("Tsakonian input").run()
        user_blocks = [m.value for m in app.chat_message[0].markdown]
        self.assertEqual(user_blocks[0], "Tsakonian input")
        self.assertIn('class="user-greek-translation"', user_blocks[1])
        self.assertIn('lang="el"', user_blocks[1])
        self.assertIn("Ελληνική μετάφραση &lt;script&gt;unsafe&lt;/script&gt;", user_blocks[1])
        self.assertNotIn("<script>", user_blocks[1])
        create.side_effect = None
        create.return_value = nim_response("Απάντηση <script>unsafe</script>")
        post.return_value = ollama_response("Tsakonian reply")
        next(b for b in app.button if b.label == "Retry failed step").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        completed_user_blocks = [m.value for m in app.chat_message[0].markdown]
        self.assertEqual(completed_user_blocks, user_blocks)
        assistant_blocks = [m.value for m in app.chat_message[1].markdown]
        self.assertEqual(assistant_blocks[0], "Tsakonian reply")
        self.assertIn('class="assistant-greek-translation"', assistant_blocks[1])
        self.assertIn('lang="el"', assistant_blocks[1])
        self.assertIn("Απάντηση &lt;script&gt;unsafe&lt;/script&gt;", assistant_blocks[1])
        self.assertNotIn("<script>", assistant_blocks[1])
        self.assertFalse(app.expander)
        self.assertFalse(app.status)

    @patch("pipeline.httpx.post")
    def test_local_assistant_receives_greek_conversation_without_translation_instructions(self, post):
        post.return_value = krikri_response()
        history = [{"ell_in": "Παλιά ερώτηση", "ell_out": "Παλιά απάντηση"}]
        self.assertEqual(ask_ollama(history, "Νέα ερώτηση", base_url="http://localhost:11434/v1/"), "Απάντηση")
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], DEFAULT_OLLAMA_MODEL)
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": GREEK_SYSTEM},
            {"role": "user", "content": "Παλιά ερώτηση"},
            {"role": "assistant", "content": "Παλιά απάντηση"},
            {"role": "user", "content": "Νέα ερώτηση"},
        ])
        self.assertNotIn("Translate from", str(payload))

    @patch("pipeline.httpx.post")
    def test_local_assistant_rejects_missing_empty_and_incomplete_answers(self, post):
        for response in [krikri_response(" "), krikri_response(done_reason="length")]:
            post.return_value = response
            with self.assertRaises(ValueError):
                ask_ollama([], "question")
        post.return_value = httpx.Response(404, request=httpx.Request("POST", "http://localhost:11434/api/chat"))
        with self.assertRaisesRegex(ValueError, "not installed"):
            ask_ollama([], "question")

    @patch.dict(os.environ, {"NVIDIA_API_KEY": ""})
    @patch("dotenv.dotenv_values", return_value={"GREEK_BACKEND": "ollama", "OLLAMA_CHAT_MODEL": DEFAULT_OLLAMA_MODEL})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_local_mode_runs_all_steps_without_an_nvidia_key(self, client_class, post, values):
        post.side_effect = [ollama_response("Ερώτηση"), krikri_response(), ollama_response("Τσακώνικα")]
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.chat_input[0].set_value("Καούρα").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        client_class.assert_not_called()
        self.assertEqual([c.kwargs["json"]["model"] for c in post.call_args_list],
                         ["leonida", DEFAULT_OLLAMA_MODEL, "leonida"])
        self.assertEqual(app.session_state["turns"][0]["greek_model"], DEFAULT_OLLAMA_MODEL)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_new_defaults_apply_without_losing_turns_and_krikri_remains_selectable(self, client_class, post):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        pending = {"tsd_in": "Καούρα", "ell_in": "Ερώτηση", "error": "Previous model timed out"}
        app.session_state["pending_turn"] = pending.copy()
        self.settings.return_value = {"GREEK_BACKEND": "ollama", "OLLAMA_CHAT_MODEL": DEFAULT_OLLAMA_MODEL}
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "ollama")
        self.assertEqual(app.selectbox[1].value, "gemma4:e4b")
        self.assertEqual(app.session_state["pending_turn"], pending)
        app.selectbox[1].select(KRIKRI_MODEL).run()
        app.run()
        self.assertEqual(app.selectbox[1].value, KRIKRI_MODEL)
        post.side_effect = [krikri_response(), ollama_response("Τσακώνικα")]
        next(b for b in app.button if b.label == "Retry failed step").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["model"], KRIKRI_MODEL)
        self.assertNotIn("think", post.call_args_list[0].kwargs["json"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(app.session_state["turns"][0]["greek_model"], KRIKRI_MODEL)
        client_class.return_value.chat.completions.create.assert_not_called()

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_failed_cloud_reply_can_resume_locally_using_saved_translation(self, client_class, post):
        nim_call = client_class.return_value.chat.completions.create
        nim_call.side_effect = APITimeoutError(request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"))
        post.return_value = ollama_response("Ερώτηση")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.chat_input[0].set_value("Καούρα").run()
        self.assertEqual(post.call_count, 1)
        app.selectbox[0].select("ollama").run()
        self.assertEqual(post.call_count, 1)
        post.side_effect = [krikri_response(), ollama_response("Τσακώνικα")]
        next(b for b in app.button if b.label == "Retry failed step").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(nim_call.call_count, 1)
        self.assertEqual(app.session_state["turns"][0]["greek_model"], DEFAULT_OLLAMA_MODEL)
        self.assertEqual(app.session_state["turns"][0]["ell_in"], "Ερώτηση")

    @patch("pipeline.httpx.post")
    def test_both_directions_use_one_model_and_exact_prompts(self, post):
        post.return_value = ollama_response()
        for origin, target, text in [
            ("Tsakonian", "Greek", "Καούρα"),
            ("Greek", "Tsakonian", "Καλημέρα"),
        ]:
            translate(text, origin, target, base_url="http://localhost:11434/v1/")
            self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/generate")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "leonida")
            self.assertEqual(payload["keep_alive"], "1h")
            self.assertTrue(payload["raw"])
            prompt = payload["prompt"]
            self.assertTrue(prompt.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"))
            self.assertIn(TRANSLATION_SYSTEM + "<|eot_id|>", prompt)
            self.assertIn(f"<|start_header_id|>user<|end_header_id|>\n\nTranslate from {origin} to {target}:\n\n{text}<|eot_id|>", prompt)
            self.assertTrue(prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
            self.assertEqual(prompt.count("Translate from "), 1)

    @patch("pipeline.httpx.post")
    def test_translation_errors_are_not_forwarded(self, post):
        for response in [ollama_response(" "), ollama_response(done_reason="length")]:
            post.return_value = response
            with self.assertRaises(ValueError):
                translate("source", "Greek", "Tsakonian")
        post.return_value = httpx.Response(
            404, request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            translate("source", "Greek", "Tsakonian")

    def test_nim_keeps_system_and_last_ten_greek_turns(self):
        client = Mock()
        client.chat.completions.create.return_value = nim_response()
        history = [{"ell_in": f"in {i}", "ell_out": f"out {i}", "tsd_in": "private"}
                   for i in range(12)]
        ask_nim(client, "nim-model", history, "current")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 22)
        self.assertEqual(messages[0], {"role": "system", "content": GREEK_SYSTEM})
        self.assertEqual(messages[1], {"role": "user", "content": "in 2"})
        self.assertEqual(messages[-1], {"role": "user", "content": "current"})
        self.assertNotIn("private", str(messages))

    def test_nim_rejects_missing_or_truncated_reply(self):
        client = Mock()
        for content, reason in [(None, "stop"), ("partial", "length")]:
            client.chat.completions.create.return_value = nim_response(content, reason)
            with self.assertRaises(ValueError):
                ask_nim(client, "nim-model", [], "question")

    def test_reasoning_model_gets_room_for_reasoning_and_bounded_length_retry(self):
        client = Mock()
        create = client.chat.completions.create
        create.side_effect = [nim_response(None, "length"), nim_response("Ολοκληρωμένη απάντηση")]
        result = ask_nim(client, "openai/gpt-oss-20b", [], "question")
        self.assertEqual(result, "Ολοκληρωμένη απάντηση")
        self.assertEqual([c.kwargs["max_tokens"] for c in create.call_args_list], [2048, 4096])
        self.assertTrue(all(c.kwargs["reasoning_effort"] == "low" for c in create.call_args_list))
        self.assertEqual(create.call_args_list[0].kwargs["messages"],
                         create.call_args_list[1].kwargs["messages"])
        self.assertEqual(create.call_count, 2)

    def test_length_retry_does_not_loop_or_retry_network_errors(self):
        client = Mock()
        create = client.chat.completions.create
        create.return_value = nim_response("unfinished", "length")
        with self.assertRaisesRegex(ValueError, "retry this step"):
            ask_nim(client, "openai/gpt-oss-20b", [], "question")
        self.assertEqual(create.call_count, 2)
        create.reset_mock()
        create.side_effect = httpx.ConnectError("Unavailable")
        with self.assertRaises(httpx.ConnectError):
            ask_nim(client, "openai/gpt-oss-20b", [], "question")
        self.assertEqual(create.call_count, 1)

    def test_other_models_do_not_receive_gpt_reasoning_options(self):
        client = Mock()
        create = client.chat.completions.create
        create.return_value = nim_response()
        ask_nim(client, "another/model", [], "question", max_tokens=1024)
        self.assertNotIn("reasoning_effort", create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 1024)
        for budget in (0, 4097):
            with self.assertRaises(ValueError):
                ask_nim(client, "another/model", [], "question", max_tokens=budget)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key", "NIM_MODEL": "old/stale-model"})
    @patch("dotenv.dotenv_values", return_value={"GREEK_BACKEND": "nim", "NIM_MODEL": DEFAULT_NIM_MODEL})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_model_setting_reloads_without_losing_pending_translation(self, client_class, post, values):
        nim_call = client_class.return_value.chat.completions.create
        nim_call.return_value = nim_response("Απάντηση")
        post.return_value = ollama_response("Τσακώνικα")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.session_state["pending_turn"] = {
            "tsd_in": "Καούρα", "ell_in": "Ερώτηση", "error": "Previous model timed out",
        }
        app.run()
        next(b for b in app.button if b.label == "Retry failed step").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(nim_call.call_args.kwargs["model"], DEFAULT_NIM_MODEL)
        self.assertNotIn("reasoning_effort", nim_call.call_args.kwargs)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(app.session_state["turns"][0]["ell_in"], "Ερώτηση")

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_cut_off_nim_reply_recovers_without_repeating_source_translation(self, client_class, post):
        nim_call = client_class.return_value.chat.completions.create
        nim_call.side_effect = [nim_response("partial", "length"), nim_response("Απάντηση")]
        post.side_effect = [ollama_response("Ερώτηση"), ollama_response("Τσακώνικα")]
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.chat_input[0].set_value("Καούρα").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(nim_call.call_count, 2)
        self.assertEqual(app.session_state["turns"][0]["ell_out"], "Απάντηση")
        self.assertIn("Απάντηση<|eot_id|>", post.call_args.kwargs["json"]["prompt"])
        self.assertNotIn("partial", post.call_args.kwargs["json"]["prompt"])

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_dictionary_branding_and_favicon(self, client_class):
        from PIL import Image
        from ui import FAVICON
        with Image.open(FAVICON) as icon:
            self.assertEqual(icon.size, (252, 252))
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        self.assertFalse(app.exception)
        markup = "\n".join(m.value for m in app.markdown)
        self.assertIn('alt="Tsakonian Digital"', markup)
        self.assertIn('aria-current="page">Digital Lab', markup)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_nim_timeout_keeps_translation_and_shows_actionable_message(self, client_class, post):
        client_class.return_value.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
        )
        post.return_value = ollama_response("Ερώτηση")
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.chat_input[0].set_value("Καούρα").run()
        self.assertFalse(app.exception)
        self.assertIn("completed steps are saved", app.error[0].value)
        self.assertEqual(app.session_state["pending_turn"]["ell_in"], "Ερώτηση")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(app.session_state["turns"], [])

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_chat_pipeline_and_failed_turn_does_not_change_history(self, client_class, post):
        client_class.return_value.chat.completions.create.return_value = nim_response("Απάντηση")
        post.side_effect = [ollama_response("Ερώτηση"), ollama_response("Τσακώνικα")]
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        self.assertFalse(app.exception)
        app.chat_input[0].set_value("Καούρα").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.session_state["turns"], [{
            "tsd_in": "Καούρα", "ell_in": "Ερώτηση", "ell_out": "Απάντηση", "tsd_out": "Τσακώνικα",
            "greek_model": DEFAULT_NIM_MODEL,
        }])
        calls = post.call_args_list
        self.assertEqual(calls[0].kwargs["json"]["model"], calls[1].kwargs["json"]["model"])
        self.assertIn("Translate from Greek to Tsakonian:\n\nΑπάντηση<|eot_id|>",
                      calls[1].kwargs["json"]["prompt"])
        post.side_effect = httpx.ConnectError("Ollama unavailable")
        app.chat_input[0].set_value("Second question").run()
        self.assertFalse(app.exception)
        self.assertIn("Ollama unavailable", app.error[0].value)
        self.assertEqual(len(app.session_state["turns"]), 1)
        self.assertEqual(app.session_state["pending_turn"]["tsd_in"], "Second question")
        next(b for b in app.button if b.label == "Clear conversation").click().run()
        self.assertEqual(app.session_state["turns"], [])
        self.assertIsNone(app.session_state["pending_turn"])

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    def test_retry_resumes_each_failed_stage_without_repeating_completed_calls(self):
        previous = {"tsd_in": "old input", "ell_in": "old Greek input",
                    "ell_out": "old Greek reply", "tsd_out": "old reply"}
        for failed_stage in (1, 2, 3):
            with self.subTest(failed_stage=failed_stage), patch("pipeline.httpx.post") as post, \
                    patch("openai.OpenAI") as client_class:
                nim_call = client_class.return_value.chat.completions.create
                error = httpx.ConnectError("Temporary outage")
                post.side_effect = ([error] if failed_stage == 1 else
                                    [ollama_response("Ερώτηση"), error] if failed_stage == 3 else
                                    [ollama_response("Ερώτηση")])
                nim_call.side_effect = error if failed_stage == 2 else None
                nim_call.return_value = nim_response("Απάντηση")
                app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
                app.session_state["turns"] = [previous.copy()]
                app.chat_input[0].set_value("Καούρα").run()
                self.assertFalse(app.exception)
                self.assertIn("Temporary outage", app.error[0].value)
                self.assertTrue(app.chat_input[0].disabled)
                self.assertEqual(app.session_state["turns"], [previous])
                pending = app.session_state["pending_turn"]
                self.assertEqual("ell_in" in pending, failed_stage > 1)
                self.assertEqual("ell_out" in pending, failed_stage > 2)

                before = (post.call_count, nim_call.call_count)
                app.run()  # A normal rerun must not silently retry a failed request.
                self.assertEqual((post.call_count, nim_call.call_count), before)
                post.side_effect = lambda *args, **kwargs: ollama_response(
                    "Ερώτηση" if "Translate from Tsakonian to Greek:" in kwargs["json"]["prompt"]
                    else "Τσακώνικα")
                nim_call.side_effect = None
                next(b for b in app.button if b.label == "Retry failed step").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertIsNone(app.session_state["pending_turn"])
                self.assertFalse(app.chat_input[0].disabled)
                self.assertEqual(app.session_state["turns"], [previous, {
                    "tsd_in": "Καούρα", "ell_in": "Ερώτηση",
                    "ell_out": "Απάντηση", "tsd_out": "Τσακώνικα",
                    "greek_model": DEFAULT_NIM_MODEL,
                }])
                self.assertEqual(post.call_count - before[0], 2 if failed_stage == 1 else 1)
                self.assertEqual(nim_call.call_count - before[1], 0 if failed_stage == 3 else 1)
                self.assertEqual(nim_call.call_args.kwargs["messages"][1:], [
                    {"role": "user", "content": previous["ell_in"]},
                    {"role": "assistant", "content": previous["ell_out"]},
                    {"role": "user", "content": "Ερώτηση"},
                ])
                calls_after_success = (post.call_count, nim_call.call_count)
                app.run()
                self.assertEqual(len(app.session_state["turns"]), 2)
                self.assertEqual((post.call_count, nim_call.call_count), calls_after_success)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
    @patch("pipeline.httpx.post")
    @patch("openai.OpenAI")
    def test_repeated_failure_keeps_results_and_discard_allows_another_message(self, client_class, post):
        nim_call = client_class.return_value.chat.completions.create
        nim_call.return_value = nim_response("Απάντηση")
        post.side_effect = [ollama_response("Ερώτηση"), httpx.ConnectError("Still unavailable")]
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
        app.chat_input[0].set_value("Καούρα").run()
        post.side_effect = httpx.ConnectError("Still unavailable")
        next(b for b in app.button if b.label == "Retry failed step").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Still unavailable", app.error[0].value)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(nim_call.call_count, 1)
        self.assertEqual(app.session_state["pending_turn"]["ell_out"], "Απάντηση")
        self.assertEqual(app.session_state["turns"], [])
        next(b for b in app.button if b.label == "Discard message").click().run()
        self.assertIsNone(app.session_state["pending_turn"])
        self.assertFalse(app.chat_input[0].disabled)
        self.assertFalse(app.error)
        post.side_effect = None
        post.return_value = ollama_response("New translation")
        app.chat_input[0].set_value("New question").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["turns"]), 1)
        self.assertEqual(app.session_state["turns"][0]["tsd_in"], "New question")


if __name__ == "__main__":
    unittest.main()
