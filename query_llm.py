import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import re
import math
import time
import timeout_decorator

from openai import OpenAI

import prompts
import utils

LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "180"))


class QueryLLM:
    def __init__(self, args):
        self.args = args
        # Load key/base_url from env first, then fallback to file for backward compatibility.
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        api_key_file = self.args.get("models", {}).get("api_key_file", "openai_key.txt")
        if not self.api_key and os.path.exists(api_key_file):
            with open(api_key_file, "r") as key_file:
                self.api_key = key_file.read().strip()
        if not self.api_key:
            raise FileNotFoundError(
                f"No API key found. Set OPENAI_API_KEY or provide key file at '{api_key_file}'."
            )

        self.api_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or self.args.get("models", {}).get("api_base_url", "")
        client_kwargs = {"api_key": self.api_key}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url
        self.client = OpenAI(**client_kwargs)
        self.is_deepseek = "deepseek" in (self.api_base_url or "").lower() or "deepseek" in self.args['models']['llm_model'].lower()
        self.use_responses_api = hasattr(self.client, "responses")
        self.use_conversations_api = hasattr(self.client, "conversations")
        self.assistants_supported = (
            hasattr(self.client, "beta")
            and hasattr(self.client.beta, "assistants")
            and hasattr(self.client.beta, "threads")
        )

        requested_mode = (
            os.getenv("OPENAI_API_MODE", "").strip().lower()
            or self.args.get("models", {}).get("api_mode", "auto")
        ).lower()
        if requested_mode not in {"auto", "assistants", "responses", "chat"}:
            print(utils.Colors.WARNING + f"Unknown api_mode={requested_mode}, fallback to auto." + utils.Colors.ENDC)
            requested_mode = "auto"

        self.use_assistants = False
        self.force_chat_only = False
        if requested_mode == "assistants":
            self.use_assistants = self.assistants_supported
            if not self.use_assistants:
                print(utils.Colors.WARNING + "Assistants mode requested but not supported by current SDK/provider. Fallback to Responses/Chat." + utils.Colors.ENDC)
        elif requested_mode == "responses":
            self.use_assistants = False
            if not self.use_responses_api:
                print(utils.Colors.WARNING + "Responses mode requested but unavailable. Fallback to Chat Completions." + utils.Colors.ENDC)
        elif requested_mode == "chat":
            self.use_assistants = False
            self.force_chat_only = True
        else:
            # Auto mode: keep previous behavior by default, but allow faster legacy Assistants path for gpt-4o when available.
            model_name = self.args['models']['llm_model'].lower()
            if model_name.startswith("gpt-4o") and self.assistants_supported:
                self.use_assistants = True

        # Deprecated branch kept as placeholders for backward compatibility.
        self.assistant = None
        self.assistant_irrelevant = None
        if self.use_assistants:
            mode_msg = "Assistants mode enabled"
            print(utils.Colors.OKBLUE + mode_msg + utils.Colors.ENDC)
        elif self.force_chat_only:
            print(utils.Colors.WARNING + "Chat Completions mode forced by api_mode=chat." + utils.Colors.ENDC)
        elif self.use_responses_api:
            mode_msg = "Responses API detected"
            if self.use_conversations_api:
                mode_msg += " with Conversations support"
            print(utils.Colors.OKBLUE + mode_msg + utils.Colors.ENDC)
        else:
            print(utils.Colors.WARNING + "Responses API not available in current openai SDK; using chat.completions fallback." + utils.Colors.ENDC)

        self.thread_irrelevant = None
        self.thread_persona = None
        self.thread_conversation = None
        self.thread_reflect_conversation = None
        self.thread_preparing_new_content = None
        self.thread_new_content = None
        self.thread_eval_new_content = None

        self.expanded_persona = ""

        self.general_personal_history = ""
        self.init_general_personal_history = ""
        self.first_expand_general_personal_history = ""
        self.second_expand_general_personal_history = ""
        self.third_expand_general_personal_history = ""

        self.init_personal_history = ""
        self.first_expand_personal_history = ""
        self.second_expand_personal_history = ""
        self.third_expand_personal_history = ""
        self.pii_profile = None

        if self.use_assistants:
            self._ensure_assistants()

    def _ensure_assistants(self):
        if not self.use_assistants:
            return
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create(
                model=self._model_for_step("expand_persona"),
                instructions="You are a helpful assistant for synthetic data generation tasks.",
            )
        if self.assistant_irrelevant is None:
            self.assistant_irrelevant = self.client.beta.assistants.create(
                model=self._model_for_step("random_question"),
                instructions="You are a helpful assistant for synthetic dialogue generation.",
            )

    @staticmethod
    def parse_synthetic_pii_from_persona_text(persona_text):
        if not isinstance(persona_text, str):
            return None
        patterns = {
            "email": r"^\s*Email:\s*(.+)\s*$",
            "phone": r"^\s*Phone:\s*(.+)\s*$",
            "id": r"^\s*ID:\s*(.+)\s*$",
            "address": r"^\s*Address:\s*(.+)\s*$",
        }
        out = {}
        for key, pat in patterns.items():
            m = re.search(pat, persona_text, flags=re.MULTILINE)
            if not m:
                return None
            out[key] = m.group(1).strip()
        return out

    def _append_pii_constraints(self, step, prompt):
        if not isinstance(self.pii_profile, dict):
            return prompt

        pii_steps = {
            "init_general_personal_history",
            "first_expand_general_personal_history",
            "second_expand_general_personal_history",
            "third_expand_general_personal_history",
            "init_contextual_personal_history",
            "first_expand_contextual_personal_history",
            "second_expand_contextual_personal_history",
            "third_expand_contextual_personal_history",
            "init_conversation",
            "first_expand_conversation",
            "second_expand_conversation",
            "third_expand_conversation",
            "expand_conversation_section",
        }
        if step not in pii_steps:
            return prompt

        email = self.pii_profile.get("email", "")
        phone = self.pii_profile.get("phone", "")
        pid = self.pii_profile.get("id", "")
        address = self.pii_profile.get("address", "")
        constraints = (
            "\n\nPII Consistency Constraints (synthetic test data only):\n"
            "- Do NOT generate new email/phone/ID/address values beyond this profile.\n"
            "- If you mention one of these fields, use the exact value from the profile.\n"
            "- Keep at most one value per field for this person across outputs.\n"
            "- If a field is not needed in context, do not force it.\n"
            f"- Synthetic profile: email={email}; phone={phone}; id={pid}; address={address}\n"
        )
        return prompt + constraints

    def _model_for_step(self, step):
        # Use the same configured model for data generation steps.
        return self.args['models']['llm_model']

    def _new_response_state(self):
        state = {"conversation_id": None, "previous_response_id": None, "messages": []}
        if self.use_responses_api and self.use_conversations_api:
            try:
                conv = self.client.conversations.create()
                state["conversation_id"] = getattr(conv, "id", None)
            except Exception:
                state["conversation_id"] = None
        return state

    @staticmethod
    def _extract_text_from_response_obj(resp):
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
        outputs = getattr(resp, "output", None)
        if isinstance(outputs, list):
            chunks = []
            for item in outputs:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        chunks.append(t)
            if chunks:
                return "\n".join(chunks)
        return ""

    @staticmethod
    def _extract_text_from_assistant_message(msg):
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            return ""
        chunks = []
        for part in content:
            if getattr(part, "type", None) != "text":
                continue
            txt = getattr(part, "text", None)
            if isinstance(txt, str):
                chunks.append(txt)
            else:
                val = getattr(txt, "value", None)
                if isinstance(val, str):
                    chunks.append(val)
        return "\n".join(chunks).strip()

    def _assistant_id_for_step(self, step):
        if not self.use_assistants:
            return None
        self._ensure_assistants()
        if step in {"random_question", "random_question_follow_up", "random_question_follow_up_response"}:
            return self.assistant_irrelevant.id
        return self.assistant.id

    def _run_assistant_turn(self, thread_id, assistant_id, prompt):
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt,
        )
        runs_api = self.client.beta.threads.runs
        if hasattr(runs_api, "create_and_poll"):
            run = runs_api.create_and_poll(thread_id=thread_id, assistant_id=assistant_id)
        else:
            run = runs_api.create(thread_id=thread_id, assistant_id=assistant_id)
            for _ in range(120):
                run = runs_api.retrieve(thread_id=thread_id, run_id=run.id)
                if getattr(run, "status", None) in {"completed", "failed", "cancelled", "expired"}:
                    break
                time.sleep(1.0)
        run_status = getattr(run, "status", None)
        if run_status != "completed":
            raise RuntimeError(f"Assistant run did not complete. status={run_status}")

        msgs = self.client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=20)
        for msg in getattr(msgs, "data", []):
            if getattr(msg, "role", None) != "assistant":
                continue
            text = self._extract_text_from_assistant_message(msg)
            if text:
                return text
        raise RuntimeError("Assistant run completed but no assistant text message found.")

    def _request_single_turn(self, model, prompt, assistant_id=None):
        if self.use_assistants and assistant_id:
            thread = self.client.beta.threads.create()
            try:
                return self._run_assistant_turn(thread_id=thread.id, assistant_id=assistant_id, prompt=prompt)
            finally:
                try:
                    self.client.beta.threads.delete(thread_id=thread.id)
                except Exception:
                    pass
        if self.use_responses_api and not self.force_chat_only:
            resp = self.client.responses.create(model=model, input=prompt)
            text = self._extract_text_from_response_obj(resp)
            if text:
                return text
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000
        )
        return response.choices[0].message.content

    def _request_with_state(self, model, prompt, state, assistant_id=None):
        if self.use_assistants and assistant_id:
            if state is None:
                state = self.client.beta.threads.create()
            thread_id = getattr(state, "id", state)
            return self._run_assistant_turn(thread_id=thread_id, assistant_id=assistant_id, prompt=prompt)

        if isinstance(state, list):
            state.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=model,
                messages=state,
                max_tokens=10000
            )
            text = completion.choices[0].message.content
            state.append({"role": "assistant", "content": text})
            return text

        if state is None:
            state = {"conversation_id": None, "previous_response_id": None, "messages": []}
        if self.use_responses_api and not self.force_chat_only:
            kwargs = {"model": model, "input": prompt}
            if state.get("previous_response_id"):
                kwargs["previous_response_id"] = state["previous_response_id"]
            if state.get("conversation_id"):
                kwargs["conversation"] = state["conversation_id"]
            try:
                resp = self.client.responses.create(**kwargs)
            except TypeError:
                kwargs.pop("conversation", None)
                resp = self.client.responses.create(**kwargs)
            state["previous_response_id"] = getattr(resp, "id", state.get("previous_response_id"))
            text = self._extract_text_from_response_obj(resp)
            if text:
                return text

        messages = state.setdefault("messages", [])
        messages.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10000
        )
        text = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": text})
        return text

    def create_a_thread(self, step):
        if self.use_assistants:
            if step == 'conversation':
                self.thread_persona = self.client.beta.threads.create()
                self.thread_conversation = self.client.beta.threads.create()
                self.thread_reflect_conversation = self.client.beta.threads.create()
            elif step == 'writing':
                self.thread_preparing_new_content = self.client.beta.threads.create()
            elif step == 'qa':
                self.thread_new_content = self.client.beta.threads.create()
                self.thread_eval_new_content = self.client.beta.threads.create()
            elif step == 'irrelevant':
                self.thread_irrelevant = self.client.beta.threads.create()
            else:
                raise ValueError(f'Invalid step: {step}')
        else:
            # Local in-memory histories to emulate threads.
            if step == 'conversation':
                if self.use_responses_api:
                    self.thread_persona = self._new_response_state()
                    self.thread_conversation = self._new_response_state()
                    self.thread_reflect_conversation = self._new_response_state()
                else:
                    self.thread_persona = []
                    self.thread_conversation = []
                    self.thread_reflect_conversation = []
            elif step == 'writing':
                self.thread_preparing_new_content = self._new_response_state() if self.use_responses_api else []
            elif step == 'qa':
                if self.use_responses_api:
                    self.thread_new_content = self._new_response_state()
                    self.thread_eval_new_content = self._new_response_state()
                else:
                    self.thread_new_content = []
                    self.thread_eval_new_content = []
            elif step == 'irrelevant':
                self.thread_irrelevant = self._new_response_state() if self.use_responses_api else []
            else:
                raise ValueError(f'Invalid step: {step}')

    def delete_a_thread(self, step):
        if self.use_assistants:
            def safe_delete(thread_id):
                try:
                    self.client.beta.threads.delete(thread_id=thread_id)
                except Exception as e:
                    print(utils.Colors.WARNING + f'Error deleting thread: {e}' + utils.Colors.ENDC)

            if step == 'conversation':
                safe_delete(thread_id=self.thread_persona.id)
                safe_delete(thread_id=self.thread_conversation.id)
                safe_delete(thread_id=self.thread_reflect_conversation.id)
            elif step == 'writing':
                safe_delete(thread_id=self.thread_preparing_new_content.id)
            elif step == 'qa':
                safe_delete(thread_id=self.thread_new_content.id)
                safe_delete(thread_id=self.thread_eval_new_content.id)
            elif step == 'irrelevant':
                safe_delete(thread_id=self.thread_irrelevant.id)
            else:
                raise ValueError(f'Invalid step: {step}')
        else:
            if step == 'conversation':
                self.thread_persona = None
                self.thread_conversation = None
                self.thread_reflect_conversation = None
            elif step == 'writing':
                self.thread_preparing_new_content = None
            elif step == 'qa':
                self.thread_new_content = None
                self.thread_eval_new_content = None
            elif step == 'irrelevant':
                self.thread_irrelevant = None
            else:
                raise ValueError(f'Invalid step: {step}')

    @timeout_decorator.timeout(LLM_TIMEOUT_SEC, timeout_exception=TimeoutError, use_signals=False)  # Thread-safe timeout mode
    def query_llm(self, step='source_data', persona=None, topic=None, seed=None, data=None, action=None, data_type=None,
                  idx_topic=0, start_time=None, verbose=False, interaction_history=None, sensitive_info_pool=None):
        schema = None
        if step == 'source_data':
            prompt = prompts.prompts_for_background_data(seed)
        elif step == 'elaborate_topic':
            prompt = prompts.prompts_for_elaborating_topic(topic)
        elif step == 'expand_persona':
            prompt = prompts.prompts_for_expanding_persona(persona, start_time)

        elif step == 'random_question':
            prompt = data + " Respond naturally and directly. "
        elif step == 'random_question_follow_up':
            prompt = prompts.prompts_for_random_question_follow_up()
        elif step == 'random_question_follow_up_response':
            prompt = data + " Respond naturally and directly. "

        elif step == 'translate_code':
            prompt = prompts.prompts_for_translating_code(data, persona)
        elif step == 'rewrite_email':
            prompt = prompts.prompts_for_rewriting_email(data, persona)
        elif step == 'rewrite_creative_writing':
            prompt = prompts.prompts_for_rewriting_creative_writing(data, persona)
        elif step == 'select_interaction_events':
            prompt = prompts.prompts_for_selecting_interaction_events(topic, data['event_history'], data['target_count'])
        elif step == 'derive_interaction_details':
            prompt = prompts.prompts_for_deriving_interaction_details(
                topic,
                data['event_record'],
                data.get('sensitive_info_pool'),
            )

        # Generate once across multiple contexts
        elif step == 'init_general_personal_history':
            prompt = prompts.prompts_for_init_general_personal_history(persona, start_time)
        elif step == 'first_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='WEEK')
        elif step == 'second_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='MONTH')
        elif step == 'third_expand_general_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(type='general', period='YEAR')

        # Generate one for each topic
        elif step == 'init_contextual_personal_history':
            prompt = prompts.prompts_for_init_contextual_personal_history(topic, start_time, self.expanded_persona, self.general_personal_history)
        elif step == 'first_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='WEEK')
        elif step == 'second_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='MONTH')
        elif step == 'third_expand_contextual_personal_history':
            prompt = prompts.prompts_for_expanding_personal_history(topic=topic, type='contextual', period='YEAR')

        # A separate thread to populate personal histories into conversations
        elif step == 'init_conversation':
            prompt = prompts.prompts_for_generating_conversations(
                topic,
                self.expanded_persona,
                curr_personal_history=self.init_personal_history,
                period='INIT',
                interaction_history=interaction_history,
                sensitive_info_pool=sensitive_info_pool,
            )
        elif step == 'first_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(
                topic,
                self.expanded_persona,
                curr_personal_history=self.first_expand_personal_history,
                period='WEEK',
                interaction_history=interaction_history,
                sensitive_info_pool=sensitive_info_pool,
            )
        elif step == 'second_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(
                topic,
                self.expanded_persona,
                curr_personal_history=self.second_expand_personal_history,
                period='MONTH',
                interaction_history=interaction_history,
                sensitive_info_pool=sensitive_info_pool,
            )
        elif step == 'third_expand_conversation':
            prompt = prompts.prompts_for_generating_conversations(
                topic,
                self.expanded_persona,
                curr_personal_history=self.third_expand_personal_history,
                period='YEAR',
                interaction_history=interaction_history,
                sensitive_info_pool=sensitive_info_pool,
            )

        # Reflect on the conversation
        elif step == 'reflect_init_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.init_personal_history, 'conversation_block': data}, round=action, period='INIT')
        elif step == 'reflect_first_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.first_expand_personal_history, 'conversation_block': data}, round=action, period='WEEK')
        elif step == 'reflect_second_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.second_expand_personal_history, 'conversation_block': data}, round=action, period='MONTH')
        elif step == 'reflect_third_expand_conversation':
            prompt = prompts.prompts_for_reflecting_conversations(topic, data={'history_block': self.third_expand_personal_history, 'conversation_block': data}, round=action, period='YEAR')

        elif step == 'expand_conversation_section':
            prompt = prompts.prompts_for_expanding_conversation_section(topic, data)

        elif step == 'qa_helper':
            prompt = prompts.prompts_for_generating_qa(data, action)
        elif step == 'prepare_new_content':
            prompt = prompts.prompt_for_preparing_new_content(data, action, data_type)
        elif step == 'new_content':
            prompt = prompts.prompt_for_content_generation(data, action)
        elif step == 'eval_new_content':
            prompt = prompts.prompt_for_evaluating_content(data, action)
        elif step == 'find_stereotype':
            prompt = prompts.prompts_for_classifying_stereotypical_preferences(data)
        else:
            raise ValueError(f'Invalid step: {step}')
        prompt = self._append_pii_constraints(step, prompt)

        # Independent API calls every time
        if (step == 'expand_persona' or step == 'qa_helper' or step == 'expand_conversation_section' or step == 'translate_code'
                or step == 'rewrite_email' or step == 'rewrite_creative_writing' or step == 'new_content' or step == 'find_stereotype'
                or step == 'select_interaction_events' or step == 'derive_interaction_details'):
            try:
                model = self._model_for_step(step)
                response = self._request_single_turn(
                    model=model,
                    prompt=prompt,
                    assistant_id=self._assistant_id_for_step(step),
                )
                if verbose:
                    print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            except Exception as e:
                raise RuntimeError(f"LLM call failed at step={step}: {e}") from None

        # API calls within a thread in a multi-turn fashion
        else:
            if step == 'source_data' or step == 'init_conversation' or step == 'first_expand_conversation' or step == 'second_expand_conversation' or step == 'third_expand_conversation':
                curr_thread = self.thread_conversation
            elif step.startswith('reflect_'):
                curr_thread = self.thread_reflect_conversation
            elif step == 'prepare_new_content':
                curr_thread = self.thread_preparing_new_content
            # elif step == 'new_content':
            #     curr_thread = self.thread_new_content
            elif step == 'eval_new_content':
                curr_thread = self.thread_eval_new_content
            elif step == 'random_question' or step == 'random_question_follow_up' or step == 'random_question_follow_up_response':
                curr_thread = self.thread_irrelevant
            else:
                curr_thread = self.thread_persona

            try:
                response = self._request_with_state(
                    model=self._model_for_step(step),
                    prompt=prompt,
                    state=curr_thread,
                    assistant_id=self._assistant_id_for_step(step),
                )
                if verbose:
                    if step == 'new_content':
                        print(f'{utils.Colors.OKGREEN}{action.capitalize()}:{utils.Colors.ENDC} {response}')
                    else:
                        print(f'{utils.Colors.OKGREEN}{topic}{utils.Colors.ENDC}' if topic else '')
                        print(f'{utils.Colors.OKGREEN}{step.capitalize()}:{utils.Colors.ENDC} {response}')
            except Exception as e:
                raise RuntimeError(f"LLM threaded call failed at step={step}: {e}") from None

        # Save general personal history to be shared across contexts
        if idx_topic == 0:
            # pattern = r'^\s*"\[(Fact|Updated Fact)\] (Likes|Dislikes)":.*$'
            # processed_response = "\n".join([line for line in response.split("\n") if not re.match(pattern, line)])
            if step == 'init_general_personal_history':
                self.general_personal_history = response
                # self.init_general_personal_history = response
            elif step == 'first_expand_general_personal_history':
                self.general_personal_history += response
                # self.first_expand_general_personal_history = response
            elif step == 'second_expand_general_personal_history':
                self.general_personal_history += response
                # self.second_expand_general_personal_history = response
            elif step == 'third_expand_general_personal_history':
                self.general_personal_history += response
                # self.third_expand_general_personal_history = response
            if step == 'expand_persona':
                self.expanded_persona = response
                parsed = self.parse_synthetic_pii_from_persona_text(response)
                if parsed:
                    self.pii_profile = parsed

        # Save general+contextual personal history in order to generate conversations
        # if step == 'init_general_personal_history':
        #     self.init_personal_history = response
        if step == 'init_contextual_personal_history':
            self.init_personal_history += response
        # elif step == 'first_expand_general_personal_history':
        #     self.first_expand_personal_history = response
        elif step == 'first_expand_contextual_personal_history':
            self.first_expand_personal_history += response
        # elif step == 'second_expand_general_personal_history':
        #     self.second_expand_personal_history = response
        elif step == 'second_expand_contextual_personal_history':
            self.second_expand_personal_history += response
        # elif step == 'third_expand_general_personal_history':
        #     self.third_expand_personal_history = response
        elif step == 'third_expand_contextual_personal_history':
            self.third_expand_personal_history += response

        return response
