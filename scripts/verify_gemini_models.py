#!/usr/bin/env python3
"""Verify access to the locked Phase 2 Gemini generator and judge models."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("❌ ERROR: No API key found in .env. Please set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    return key.strip()


def check_model_rest(api_key: str, model_name: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    generation_config = {
        "maxOutputTokens": 2048 if model_name.lower().startswith("gemini-3.7") else 32
    }
    if not model_name.lower().startswith("gemini-3.7"):
        generation_config["temperature"] = 0.0
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Respond with exactly: 'OK'"}
                ]
            }
        ],
        "generationConfig": generation_config,
    }
    start = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, timeout=45.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            text = ""
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
            
            usage = data.get("usageMetadata", {})
            return {
                "model": model_name,
                "status": "PASS",
                "status_code": 200,
                "latency_ms": round(elapsed_ms, 2),
                "response": text,
                "prompt_tokens": usage.get("promptTokenCount"),
                "candidates_tokens": usage.get("candidatesTokenCount"),
                "total_tokens": usage.get("totalTokenCount"),
            }
        else:
            return {
                "model": model_name,
                "status": "FAIL",
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "error": response.text[:200],
            }
    except Exception as e:
        return {
            "model": model_name,
            "status": "ERROR",
            "status_code": 0,
            "latency_ms": 0,
            "error": str(e),
        }


def check_openai_compat(api_key: str, model_name: str) -> dict:
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Respond with: 'OpenAI-compat OK'"}],
        "max_tokens": 2048 if model_name.lower().startswith("gemini-3.7") else 32,
    }
    if not model_name.lower().startswith("gemini-3.7"):
        payload["temperature"] = 0.0
    start = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=45.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            return {
                "model": model_name,
                "status": "PASS",
                "latency_ms": round(elapsed_ms, 2),
                "response": text,
                "usage": usage
            }
        else:
            return {
                "model": model_name,
                "status": "FAIL",
                "status_code": response.status_code,
                "error": response.text[:200]
            }
    except Exception as e:
        return {
            "model": model_name,
            "status": "ERROR",
            "error": str(e)
        }


def list_available_models(api_key: str) -> list[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = httpx.get(url, timeout=15.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"].replace("models/", "") for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception:
        pass
    return []


def main():
    print("==================================================================")
    print("  GEMINI API & MODEL ACCESS VERIFICATION TOOL")
    print("==================================================================")
    
    api_key = get_api_key()
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"🔑 API Key Detected: {masked_key}\n")

    print("🔍 Fetching available models list from Google AI Studio...")
    models = list_available_models(api_key)
    if models:
        gemini_models = [m for m in models if "gemini" in m]
        print(f"✅ Found {len(gemini_models)} Gemini models available in your account:")
        for m in sorted(gemini_models):
            print(f"   • {m}")
    else:
        print("⚠️ Could not fetch model list directly, testing target models individually.")

    targets = [
        ("gemini-3.1-flash-lite", "Generator"),
        ("gemini-3.7-flash", "RAGAS judge"),
    ]

    print("\n------------------------------------------------------------------")
    print("  TESTING DIRECT GOOGLE AI REST API CONTRACTS")
    print("------------------------------------------------------------------")

    for model_name, role in targets:
        result = check_model_rest(api_key, model_name)
        if result["status"] == "PASS":
            print(f"✅ [{role}] {model_name:20} -> PASS (Latency: {result['latency_ms']} ms | Tokens: {result['total_tokens']})")
        else:
            print(f"❌ [{role}] {model_name:20} -> {result['status']} (Code: {result.get('status_code')}, Error: {result.get('error')})")

    print("\n------------------------------------------------------------------")
    print("  TESTING OPENAI-COMPATIBLE ENDPOINT (/v1beta/openai/)")
    print("------------------------------------------------------------------")

    for model_name, role in targets:
        result = check_openai_compat(api_key, model_name)
        if result["status"] == "PASS":
            print(f"✅ [{role}] {model_name:20} -> PASS (Response: '{result['response']}', Latency: {result['latency_ms']} ms)")
        else:
            print(f"❌ [{role}] {model_name:20} -> {result['status']} ({result.get('error')})")

    print("\n==================================================================")
    print("  PHASE 2 MODEL LOCK")
    print("==================================================================")
    print("  Generator : gemini-3.1-flash-lite")
    print("  RAGAS judge: gemini-3.7-flash")
    print("  Read current project quota from Google AI Studio before a full run.")
    print("==================================================================")


if __name__ == "__main__":
    main()
