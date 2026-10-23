import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

import aliyun_asr as cloud
import numpy as np


class AliyunTests(unittest.TestCase):
    def test_unknown_speaker_never_inherits_previous_identity(self):
        payload = {"transcripts": [{"sentences": [
            {"begin_time": 3000, "end_time": 4000, "speaker_id": 71, "text": "Again."},
            {"begin_time": 0, "end_time": 500, "speaker_id": 71, "text": "Hello."},
            {"begin_time": 700, "end_time": 900, "text": "Unknown."},
            {"begin_time": 1000, "end_time": 2000, "speaker_id": 4, "text": "Other."},
        ]}]}
        lines, count, unknown = cloud.parse_result(payload)
        self.assertEqual([x["speaker"] for x in lines], ["说话人 A", "说话人未判定", "说话人 B", "说话人 A"])
        self.assertEqual(lines[-1]["start"], 3.0)
        self.assertEqual((count, unknown), (2, 1))

    @patch("aliyun_asr.time.sleep")
    @patch("aliyun_asr.request_json")
    def test_async_request_uses_documented_diarization_fields(self, request, sleep):
        request.side_effect = [
            {"output": {"task_id": "test-task"}},
            {"output": {"task_status": "PENDING"}},
            {"output": {"task_status": "SUCCEEDED", "results": [
                {"subtask_status": "SUCCEEDED", "transcription_url": "https://results.oss-cn-beijing.aliyuncs.com/result.json"}
            ]}},
            {"transcripts": []},
        ]
        config = {"base": "https://dashscope.aliyuncs.com/api/v1", "model": "paraformer-v2", "api_key": "secret"}
        payload, task = cloud.submit_and_wait(config, "https://private.example/audio?signature=x", 3, lambda *args: None)
        submit = request.call_args_list[0].args[1]
        self.assertTrue(submit["parameters"]["diarization_enabled"])
        self.assertEqual(submit["parameters"]["speaker_count"], 3)
        self.assertNotIn("api_key", request.call_args_list[-1].kwargs)
        self.assertEqual(task, "test-task")

    @patch("aliyun_asr.request_json")
    def test_subtask_failure_is_not_treated_as_success(self, request):
        request.side_effect = [
            {"output": {"task_id": "test"}},
            {"output": {"task_status": "SUCCEEDED", "results": [
                {"subtask_status": "FAILED", "code": "DOWNLOAD_FAILED", "file_url": "https://private/?secret=key"}
            ]}},
        ]
        with self.assertRaisesRegex(cloud.CloudASRError, "DOWNLOAD_FAILED") as error:
            cloud.submit_and_wait({"base": "https://dashscope.aliyuncs.com/api/v1", "api_key": "secret", "model": "paraformer-v2"}, "url", 3, lambda *args: None)
        self.assertNotIn("private", str(error.exception))

    @patch("aliyun_asr.urlopen")
    def test_http_errors_do_not_expose_api_key_or_signed_url(self, urlopen):
        urlopen.side_effect = HTTPError("https://signed/?Signature=secret", 401, "Unauthorized", {},
                                       io.BytesIO(json.dumps({"code": "InvalidApiKey", "message": "echo secret"}).encode()))
        with self.assertRaises(cloud.CloudASRError) as error:
            cloud.request_json("https://dashscope.aliyuncs.com/api/v1/tasks/test", api_key="secret")
        self.assertIn("401", str(error.exception))
        self.assertNotIn("secret", str(error.exception))

    def test_uploaded_object_and_local_audio_are_deleted_after_failure(self):
        bucket = MagicMock()
        bucket.sign_url.return_value = "https://private.example/audio?signature=secret"
        sdk = types.SimpleNamespace(Auth=MagicMock(), Bucket=MagicMock(return_value=bucket))
        config = {"access_id": "id", "access_secret": "secret", "endpoint": "https://oss-cn-beijing.aliyuncs.com", "bucket": "example-bucket"}
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "test.mp4"
            with patch.dict("sys.modules", {"oss2": sdk}), patch.object(cloud, "settings", return_value=config), \
                    patch.object(cloud, "submit_and_wait", side_effect=cloud.CloudASRError("failed")):
                with self.assertRaises(cloud.CloudASRError):
                    cloud.transcribe(media, 3, lambda path: np.zeros(16000, dtype=np.float32))
            bucket.delete_object.assert_called_once()
            self.assertEqual(bucket.put_object_from_file.call_args.args[0], bucket.delete_object.call_args.args[0])
            self.assertFalse(media.with_suffix(".cloud.wav").exists())

    def test_workspace_url_is_normalized_without_openai_compatible_path(self):
        env = {"DASHSCOPE_API_KEY": "test", "ALIYUN_OSS_ACCESS_KEY_ID": "test", "ALIYUN_OSS_ACCESS_KEY_SECRET": "test",
               "ALIYUN_OSS_BUCKET": "test-bucket", "ALIYUN_OSS_ENDPOINT": "oss-cn-guangzhou.aliyuncs.com",
               "DASHSCOPE_BASE_URL": "https://example.cn-beijing.maas.aliyuncs.com/compatible/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = cloud.settings()
        self.assertEqual(config["base"], "https://example.cn-beijing.maas.aliyuncs.com/api/v1")


if __name__ == "__main__":
    unittest.main()
