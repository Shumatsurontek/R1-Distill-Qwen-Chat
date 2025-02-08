import unittest
import torch
import subprocess
import requests
import time
import os

class TestVLLMCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Démarrer le serveur vLLM en arrière-plan
        cls.vllm_process = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", "0.85",
            "--dtype", "bfloat16"
        ])
        # Attendre que le serveur démarre
        time.sleep(30)

    @classmethod
    def tearDownClass(cls):
        # Arrêter le serveur
        cls.vllm_process.terminate()
        cls.vllm_process.wait()

    def test_cuda_available(self):
        """Vérifier que CUDA est disponible"""
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 1)

    def test_cuda_version(self):
        """Vérifier la version CUDA"""
        version = torch.version.cuda
        self.assertIsNotNone(version)
        self.assertEqual(version.split('.')[0], '12')  # CUDA 12.x

    def test_gpu_memory(self):
        """Vérifier la mémoire GPU disponible"""
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory
            self.assertGreaterEqual(free_memory, 8 * 1024 * 1024 * 1024)  # Au moins 8GB

    def test_vllm_api_health(self):
        """Vérifier que l'API vLLM répond"""
        try:
            response = requests.get('http://localhost:8000/health')
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.RequestException as e:
            self.fail(f"L'API vLLM ne répond pas: {e}")

    def test_vllm_inference(self):
        """Tester une inférence simple"""
        try:
            response = requests.post(
                'http://localhost:8000/v1/chat/completions',
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    "messages": [{"role": "user", "content": "Dis bonjour"}],
                    "max_tokens": 50
                }
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn('choices', response.json())
        except requests.exceptions.RequestException as e:
            self.fail(f"Erreur lors de l'inférence: {e}")

    def test_gpu_compute_capability(self):
        """Vérifier la capacité de calcul GPU"""
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            self.assertGreaterEqual(capability[0], 7)  # RTX 4070 devrait avoir une capacité > 7.0 