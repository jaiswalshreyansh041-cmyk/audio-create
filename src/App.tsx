/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { Upload, FileAudio, Activity, AlertTriangle, CheckCircle, BrainCircuit, Settings2, Download, Shield, MessageSquare } from 'lucide-react';
import { GoogleGenAI } from '@google/genai';

interface AudioStats {
  format: string;
  sampleRate: number;
  numChannels: number;
  duration: number;
  peakDB: number;
  rmsDB: number;
  clipCount: number;
  dcOffset: number;
  noiseLevelDB: number;
  snrDB: number;
  silenceRatio: number;
  maxSilenceDuration: number;
  rmsDynamicRange: number;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [stats, setStats] = useState<AudioStats | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [aiResult, setAiResult] = useState<string | null>(null);
  const [aiJsonResult, setAiJsonResult] = useState<string | null>(null);
  const [isAiAnalyzing, setIsAiAnalyzing] = useState(false);
  const [isGeneratingJson, setIsGeneratingJson] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const noiseThreshold = -45;
  const snrThreshold = 20;
  const maxSilenceRatio = 20;
  const expectedSpeakers = 2;
  const minDuration = 30;
  const maxDuration = 300;

  const audioRef = useRef<HTMLAudioElement>(null);

  let parsedAiData: any = null;
  if (aiJsonResult) {
    try {
      parsedAiData = JSON.parse(aiJsonResult);
    } catch (e) {
      console.error("Failed to parse AI JSON result", e);
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setStats(null);
    setError(null);
    setAiResult(null);
    setAiError(null);
    setIsAnalyzing(true);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      const numChannels = audioBuffer.numberOfChannels;
      const sampleRate = audioBuffer.sampleRate;
      const duration = audioBuffer.duration;

      // Mixdown to mono for easier analysis
      const monoData = new Float32Array(audioBuffer.length);
      for (let i = 0; i < audioBuffer.length; i++) {
        let sum = 0;
        for (let c = 0; c < numChannels; c++) {
          sum += audioBuffer.getChannelData(c)[i];
        }
        monoData[i] = sum / numChannels;
      }

      let peak = 0;
      let sumSquares = 0;
      let clipCount = 0;
      let dcOffsetSum = 0;
      let silenceSamples = 0;
      const silenceThreshold = 0.001; // approx -60 dBFS

      let currentSilenceSamples = 0;
      let maxSilenceSamples = 0;

      const frameSize = Math.floor(sampleRate * 0.05); // 50ms frames
      const rmsValues: number[] = [];
      
      const windowSize = sampleRate; // 1 second window for dynamic range
      let windowRmsSum = 0;
      let windowSamples = 0;
      const rmsWindows: number[] = [];

      for (let i = 0; i < monoData.length; i++) {
        const sample = monoData[i];
        const absSample = Math.abs(sample);

        if (absSample > peak) peak = absSample;
        sumSquares += sample * sample;
        dcOffsetSum += sample;

        // 1. Clipping detection (audio exceeds or hits 0 dBFS -> 1.0)
        if (absSample >= 0.999) clipCount++;

        if (absSample < silenceThreshold) {
          silenceSamples++;
          currentSilenceSamples++;
          if (currentSilenceSamples > maxSilenceSamples) {
            maxSilenceSamples = currentSilenceSamples;
          }
        } else {
          currentSilenceSamples = 0;
        }

        windowRmsSum += sample * sample;
        windowSamples++;
        if (windowSamples >= windowSize) {
          const windowRms = Math.sqrt(windowRmsSum / windowSize);
          if (windowRms > 0.001) {
            rmsWindows.push(20 * Math.log10(windowRms));
          }
          windowRmsSum = 0;
          windowSamples = 0;
        }
      }

      for (let i = 0; i < monoData.length; i += frameSize) {
        let frameSumSq = 0;
        let count = 0;
        for (let j = 0; j < frameSize && i + j < monoData.length; j++) {
          frameSumSq += monoData[i + j] * monoData[i + j];
          count++;
        }
        rmsValues.push(Math.sqrt(frameSumSq / count));
      }

      rmsValues.sort((a, b) => a - b);
      // 2. Background noise level (10th percentile of frame RMS)
      const noiseRMS = rmsValues[Math.floor(rmsValues.length * 0.1)] || 0.00001;
      // Signal level (90th percentile of frame RMS)
      const signalRMS = rmsValues[Math.floor(rmsValues.length * 0.9)] || 0.00001;

      const noiseLevelDB = 20 * Math.log10(noiseRMS);
      const signalLevelDB = 20 * Math.log10(signalRMS);
      
      // 3. Signal-to-Noise Ratio
      const snrDB = signalLevelDB - noiseLevelDB;

      // 4. Silence ratio
      const silenceRatio = silenceSamples / monoData.length;
      const maxSilenceDuration = maxSilenceSamples / sampleRate;

      let rmsDynamicRange = 0;
      if (rmsWindows.length > 0) {
        rmsDynamicRange = Math.max(...rmsWindows) - Math.min(...rmsWindows);
      }

      const rms = Math.sqrt(sumSquares / monoData.length);
      const peakDB = peak > 0 ? 20 * Math.log10(peak) : -Infinity;
      const rmsDB = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
      const dcOffset = dcOffsetSum / monoData.length;

      setStats({
        format: selectedFile.type || selectedFile.name.split('.').pop()?.toUpperCase() || 'Unknown',
        sampleRate,
        numChannels,
        duration,
        peakDB,
        rmsDB,
        clipCount,
        dcOffset,
        noiseLevelDB,
        snrDB,
        silenceRatio,
        maxSilenceDuration,
        rmsDynamicRange
      });
    } catch (err) {
      console.error(err);
      setError("Failed to analyze audio. The file might be corrupted or unsupported.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
  };

  const runAiAnalysis = async () => {
    if (!file) return;
    setIsAiAnalyzing(true);
    setAiError(null);
    setAiResult(null); 
    setAiJsonResult(null);

    try {
      // Fetch runtime API key from backend
      const configRes = await fetch('/api/config');
      if (!configRes.ok) throw new Error("Failed to fetch API configuration");
      const config = await configRes.json();
      
      const apiKey = config.GEMINI_API_KEY;
      const transcriptApiKey = config.TRANSCRIPT_API_KEY || apiKey; // Fallback to primary if not set
      const jsonApiKey = config.JSON_API_KEY || apiKey; // Fallback to primary if not set
      if (!apiKey && !jsonApiKey) {
        throw new Error("API key is missing from environment variables.");
      }

      const withRetry = async <T,>(fn: () => Promise<T>, maxRetries = 4, baseDelay = 2000): Promise<T> => {
        let attempt = 0;
        while (attempt < maxRetries) {
          try {
            return await fn();
          } catch (error: any) {
            const isRetryable = error?.status === 503 || error?.message?.includes('503') || error?.message?.includes('high demand') || error?.status === 429;
            if (isRetryable && attempt < maxRetries - 1) {
              attempt++;
              const delay = baseDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
              console.warn(`API error (503/429). Retrying attempt ${attempt} in ${Math.round(delay)}ms...`);
              await new Promise(resolve => setTimeout(resolve, delay));
            } else {
              throw error;
            }
          }
        }
        throw new Error("Max retries reached");
      };

      // 1. Upload file using Gemini SDK using the Analysis key
      const aiAnalysis = new GoogleGenAI({ apiKey: jsonApiKey });
      const uploadResultAnalysis = await withRetry(() => aiAnalysis.files.upload({
        file: file,
        config: {
          mimeType: file.type || 'audio/mp3',
          displayName: `analysis-${file.name}`
        }
      }));
      const fileUriAnalysis = uploadResultAnalysis.uri;

      // 1b. Upload file using Gemini SDK using the Transcript key
      const aiTranscript = new GoogleGenAI({ apiKey: transcriptApiKey });
      const uploadResultTranscript = await withRetry(() => aiTranscript.files.upload({
        file: file,
        config: {
          mimeType: file.type || 'audio/mp3',
          displayName: `transcript-${file.name}`
        }
      }));
      const fileUriTranscript = uploadResultTranscript.uri;

      // 2. Generate JSON Analysis directly
      setIsGeneratingJson(true);
      try {
        const analysisPromise = withRetry(() => aiAnalysis.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: [
            {
              fileData: {
                fileUri: fileUriAnalysis,
                mimeType: file.type || 'audio/mp3'
              }
            },
            `Analyze the audio and provide a JSON response with ONLY the following structure. Do NOT include transcripts. ALL SCORES MUST BE NUMBERS (e.g. 0.5), NOT STRINGS:
{
  "content_safety_scores": {
    "toxicity_detected": "Number 0.0 to 1.0 (Reject >= 0.3)",
    "sexual_content_detected": "Number 0.0 to 1.0 (Reject >= 0.2)",
    "violent_intent": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "political_campaigning": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "discriminatory_content": "Number 0.0 to 1.0 (Reject >= 0.3)",
    "pii_leakage": "Number (Count of real PII entities like phone, Aadhaar, PAN. Reject if > 0)"
  },
  "conversation_quality_metrics": {
    "is_scripted": "Number 0.0 to 1.0 (Reject >= 0.85)",
    "real_world_impersonation": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "excessive_code_switching": "Number 0.0 to 1.0 (Reject > 0.3)",
    "topic_coherence_fail": "Number 0.0 to 1.0 (Reject >= 0.75)",
    "native_speaker_fail": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "task_alignment_fail": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "emotion_sentiment_mismatch": "Number 0.0 to 1.0 (Reject >= 0.5)"
  },
  "voice_quality_metrics": {
    "unnatural_pauses": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "robotic_tone": "Number 0.0 to 1.0 (Reject >= 0.5)",
    "audio_glitches": "Number 0.0 to 1.0 (Reject >= 0.5)"
  }
}`
          ],
          config: { responseMimeType: "application/json" }
        }));

        const transcriptPromise = withRetry(() => aiTranscript.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: [
            {
              fileData: {
                fileUri: fileUriTranscript,
                mimeType: file.type || 'audio/mp3'
              }
            },
            `Generate a transcript of this audio in JSON format exactly like this:
{
  "speakers": ["Speaker 1", "Speaker 2"],
  "transcript_by_turn": [
    {
      "speaker": "Speaker 1",
      "start_time": "[MM:SS]",
      "end_time": "[MM:SS]",
      "text": "Exact verbatim text written in the NATIVE SCRIPT of the spoken language (e.g. Telugu: తెలుగు లిపి, Hindi: देवनागरी, Tamil: தமிழ் எழுத்து). Do NOT romanize or transliterate — always use the native Unicode script."
    }
  ]
}
**TRANSCRIPTION RULES:**
- Native Script: ALWAYS write the spoken words in their native Unicode script. Telugu → తెలుగు లిపి, Hindi → देवनागरी, Tamil → தமிழ், Kannada → ಕನ್ನಡ, etc. NEVER use Roman/Latin transliteration.
- English words spoken within a native-language sentence: keep them in English as spoken.
- Beeps / Sensitive Info: If a beep replaces PII (name, DOB, phone), write [beep]. Never guess the hidden info.
- Overlapping / Interruptions: If two speakers talk at the same time, write each speaker on a separate line with their respective timestamps. Do NOT add notes like (overlapping) or (interruption).
- Fillers: Keep natural fillers exactly as spoken, written in native script.
- Cut-off Sentences: If a speaker is interrupted, end with a dash —.
- Accuracy: Write exactly what you hear. No paraphrasing or commentary.
- Speaker Labels: Always use Speaker 1, Speaker 2, etc. Each speaker always starts on a new line.`
          ],
          config: { responseMimeType: "application/json" }
        }));

        const [analysisSettled, transcriptSettled] = await Promise.allSettled([analysisPromise, transcriptPromise]);

        let aiDataObj = {};
        let transcriptDataObj = {};

        if (analysisSettled.status === 'fulfilled') {
          try {
            const cleanAnalysis = (analysisSettled.value.text || "{}").replace(/```json/gi, '').replace(/```/g, '').trim();
            aiDataObj = JSON.parse(cleanAnalysis);
          } catch (e) {
            console.error("Analysis JSON parse error:", e, analysisSettled.value.text);
          }
        } else {
          console.error("Analysis API call failed:", analysisSettled.reason);
        }

        if (transcriptSettled.status === 'fulfilled') {
          try {
            const cleanTranscript = (transcriptSettled.value.text || "{}").replace(/```json/gi, '').replace(/```/g, '').trim();
            transcriptDataObj = JSON.parse(cleanTranscript);
          } catch (e) {
            console.error("Transcript JSON parse error:", e, transcriptSettled.value.text);
          }
        } else {
          console.error("Transcript API call failed:", transcriptSettled.reason);
        }

        const aiData = { ...aiDataObj, ...transcriptDataObj };

        const programmaticData = {
          duration_seconds: Number(stats?.duration.toFixed(2) || 0),
          snr_db: Number(stats?.snrDB.toFixed(2) || 0),
          clipping_detected: stats?.clipCount ? stats.clipCount > 0 : false,
          background_noise_dbfs: Number(stats?.noiseLevelDB.toFixed(2) || 0),
          dc_offset: Number(stats?.dcOffset.toFixed(6) || 0),
          silence_ratio_percent: Number(((stats?.silenceRatio || 0) * 100).toFixed(2)),
          prosody_per_turn: { pitch_hz: null, intensity_db: null },
          diarization_rttm: null
        };

        const masterJson = {
          audio_metrics: programmaticData,
          ai_analysis: aiData
        };

        setAiJsonResult(JSON.stringify(masterJson, null, 2));
      } catch (err) {
        console.error("JSON Generation Error:", err);
        throw err;
      } finally {
        setIsGeneratingJson(false);
      }

    } catch (err: any) {
      console.error(err);
      let errorMessage = err.message || "Failed to run AI analysis. Please check your API key and try again.";
      try {
        const errorJson = JSON.parse(errorMessage);
        if (errorJson.error && errorJson.error.message) {
          errorMessage = errorJson.error.message;
        }
      } catch(e) {
        // If it's not valid JSON, just keep the original error message
      }
      setAiError(errorMessage);
    } finally {
      setIsAiAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            Audio Quality-Control Dashboard
          </h1>
          <p className="text-neutral-500 text-lg">
            Upload an audio file to run deterministic quality checks and AI-powered semantic analysis.
          </p>
        </header>

        <section className="bg-white p-8 rounded-2xl shadow-sm border border-neutral-200">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-neutral-300 rounded-xl p-12 bg-neutral-50 hover:bg-neutral-100 transition-colors cursor-pointer relative">
            <input 
              type="file" 
              accept="audio/wav,audio/mpeg,audio/flac,audio/*" 
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="w-12 h-12 text-neutral-400 mb-4" />
            <p className="text-lg font-medium text-neutral-700">Click or drag to upload audio</p>
            <p className="text-neutral-500 text-sm mt-1">Supports WAV, MP3, FLAC</p>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <p>{error}</p>
            </div>
          )}

          {isAnalyzing && (
            <div className="mt-6 flex items-center justify-center gap-3 text-neutral-500">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
              Analyzing audio waveform...
            </div>
          )}

          {file && !isAnalyzing && !error && (
            <div className="mt-6">
              <audio ref={audioRef} src={URL.createObjectURL(file)} controls className="w-full" />
            </div>
          )}
        </section>

        {stats && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <section>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <FileAudio className="w-5 h-5 text-neutral-500" />
                Metadata & Format
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Format" value={stats.format} />
                <MetricCard label="Sample Rate" value={`${stats.sampleRate} Hz`} />
                <MetricCard label="Channels" value={stats.numChannels === 1 ? 'Mono' : stats.numChannels === 2 ? 'Stereo' : stats.numChannels.toString()} />
                <MetricCard label="Duration" value={formatTime(stats.duration)} />
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-neutral-500" />
                Functional Quality Checks
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatusCard 
                  label="Clipping (0 dBFS)" 
                  isWarning={stats.clipCount > 0}
                  value={stats.clipCount > 0 ? `Detected (${stats.clipCount} samples)` : 'No Clipping'}
                />
                
                <StatusCard 
                  label="Background Noise" 
                  isWarning={stats.noiseLevelDB > noiseThreshold}
                  value={`${stats.noiseLevelDB.toFixed(1)} dBFS`}
                />

                <StatusCard 
                  label="Signal-to-Noise Ratio" 
                  isWarning={stats.snrDB < snrThreshold}
                  value={`${stats.snrDB.toFixed(1)} dB`}
                />

                <StatusCard 
                  label="Silence Ratio" 
                  isWarning={(stats.silenceRatio * 100) > maxSilenceRatio}
                  value={`${(stats.silenceRatio * 100).toFixed(1)}%`}
                />

                <StatusCard 
                  label="Max Silence Duration" 
                  isWarning={false}
                  value={`${stats.maxSilenceDuration.toFixed(2)}s`}
                />

                <StatusCard 
                  label="RMS Dynamic Range" 
                  isWarning={false}
                  value={`${stats.rmsDynamicRange.toFixed(1)} dB`}
                />

                <StatusCard 
                  label="Duration Check" 
                  isWarning={stats.duration < minDuration || stats.duration > maxDuration}
                  value={stats.duration < minDuration ? 'Too Short' : stats.duration > maxDuration ? 'Too Long' : 'Passed'}
                />

                <StatusCard 
                  label="Speaker Count" 
                  isWarning={parsedAiData?.ai_analysis?.speakers ? parsedAiData.ai_analysis.speakers.length !== expectedSpeakers : false}
                  value={parsedAiData?.ai_analysis?.speakers ? `${parsedAiData.ai_analysis.speakers.length} (Expected ${expectedSpeakers})` : 'Pending AI'}
                />
              </div>
              
              {Math.abs(stats.dcOffset) > 0.01 && (
                <div className="mt-4 p-4 bg-amber-50 text-amber-800 rounded-lg flex items-start gap-3 border border-amber-200">
                  <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-amber-600" />
                  <div>
                    <p className="font-medium">Significant DC Offset Detected</p>
                    <p className="text-sm mt-1">The waveform is not centered at zero (Offset: {stats.dcOffset.toFixed(4)}). This can reduce headroom and cause clicks during editing.</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.content_safety_scores ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Shield className="w-5 h-5 text-neutral-500" />
                    Content Safety
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Toxicity Detected" value={parsedAiData.ai_analysis.content_safety_scores.toxicity_detected || 0} threshold=">= 0.3" isWarning={(parsedAiData.ai_analysis.content_safety_scores.toxicity_detected || 0) >= 0.3} />
                    <ScoreCard label="Sexual Content Detected" value={parsedAiData.ai_analysis.content_safety_scores.sexual_content_detected || 0} threshold=">= 0.2" isWarning={(parsedAiData.ai_analysis.content_safety_scores.sexual_content_detected || 0) >= 0.2} />
                    <ScoreCard label="Violent Intent" value={parsedAiData.ai_analysis.content_safety_scores.violent_intent || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.content_safety_scores.violent_intent || 0) >= 0.5} />
                    <ScoreCard label="Political Campaigning" value={parsedAiData.ai_analysis.content_safety_scores.political_campaigning || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.content_safety_scores.political_campaigning || 0) >= 0.5} />
                    <ScoreCard label="Discriminatory Content" value={parsedAiData.ai_analysis.content_safety_scores.discriminatory_content || 0} threshold=">= 0.3" isWarning={(parsedAiData.ai_analysis.content_safety_scores.discriminatory_content || 0) >= 0.3} />
                    <ScoreCard label="PII Leakage" value={parsedAiData.ai_analysis.content_safety_scores.pii_leakage || 0} threshold="> 0" isWarning={(parsedAiData.ai_analysis.content_safety_scores.pii_leakage || 0) > 0} isInteger={true} />
                  </div>
                </div>
              ) : (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Shield className="w-5 h-5 text-neutral-500" />
                    Content Safety
                  </h3>
                  <div className="bg-neutral-50 border border-neutral-200 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-neutral-500">
                    <BrainCircuit className="w-8 h-8 mb-2 text-neutral-400" />
                    <p>Click "Run AI Analysis" below to run AI-powered safety checks.</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.conversation_quality_metrics ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Conversation Quality
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Is Scripted" value={parsedAiData.ai_analysis.conversation_quality_metrics.is_scripted || 0} threshold=">= 0.85" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.is_scripted || 0) >= 0.85} />
                    <ScoreCard label="Real World Impersonation" value={parsedAiData.ai_analysis.conversation_quality_metrics.real_world_impersonation || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.real_world_impersonation || 0) >= 0.5} />
                    <ScoreCard label="Excessive Code Switching" value={parsedAiData.ai_analysis.conversation_quality_metrics.excessive_code_switching || 0} threshold="> 0.3" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.excessive_code_switching || 0) > 0.3} />
                    <ScoreCard label="Topic Coherence Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.topic_coherence_fail || 0} threshold=">= 0.75" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.topic_coherence_fail || 0) >= 0.75} />
                    <ScoreCard label="Native Speaker Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.native_speaker_fail || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.native_speaker_fail || 0) >= 0.5} />
                    <ScoreCard label="Task Alignment Fail" value={parsedAiData.ai_analysis.conversation_quality_metrics.task_alignment_fail || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.task_alignment_fail || 0) >= 0.5} />
                    <ScoreCard label="Emotion Sentiment Mismatch" value={parsedAiData.ai_analysis.conversation_quality_metrics.emotion_sentiment_mismatch || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.conversation_quality_metrics.emotion_sentiment_mismatch || 0) >= 0.5} />
                  </div>
                </div>
              ) : (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Conversation Quality
                  </h3>
                  <div className="bg-neutral-50 border border-neutral-200 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-neutral-500">
                    <BrainCircuit className="w-8 h-8 mb-2 text-neutral-400" />
                    <p>Click "Run AI Analysis" below to run AI-powered quality checks.</p>
                  </div>
                </div>
              )}

              {parsedAiData?.ai_analysis?.voice_quality_metrics ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Activity className="w-5 h-5 text-neutral-500" />
                    Voice Quality
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <ScoreCard label="Unnatural Pauses" value={parsedAiData.ai_analysis.voice_quality_metrics.unnatural_pauses || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.unnatural_pauses || 0) >= 0.5} />
                    <ScoreCard label="Robotic Tone" value={parsedAiData.ai_analysis.voice_quality_metrics.robotic_tone || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.robotic_tone || 0) >= 0.5} />
                    <ScoreCard label="Audio Glitches" value={parsedAiData.ai_analysis.voice_quality_metrics.audio_glitches || 0} threshold=">= 0.5" isWarning={(parsedAiData.ai_analysis.voice_quality_metrics.audio_glitches || 0) >= 0.5} />
                  </div>
                </div>
              ) : null}

              {parsedAiData?.ai_analysis?.transcript_by_turn ? (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <MessageSquare className="w-5 h-5 text-neutral-500" />
                    Transcript
                  </h3>
                  <div className="bg-white p-6 rounded-xl border border-neutral-200 shadow-sm space-y-4">
                    {parsedAiData.ai_analysis.transcript_by_turn.map((turn: any, idx: number) => (
                      <div key={idx} className="flex flex-col sm:flex-row gap-2 sm:gap-4">
                        <div className="sm:w-32 shrink-0">
                          <span className="font-semibold text-blue-700">{turn.speaker}</span>
                          <div className="text-xs text-neutral-500">{turn.start_time} - {turn.end_time}</div>
                        </div>
                        <div className="text-neutral-800">
                          {turn.text}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {aiJsonResult && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-neutral-700">
                    <Settings2 className="w-5 h-5 text-neutral-500" />
                    Raw JSON Output
                  </h3>
                  <div className="bg-neutral-900 text-neutral-100 p-6 rounded-xl overflow-x-auto text-sm font-mono shadow-inner">
                    <pre>{aiJsonResult}</pre>
                  </div>
                </div>
              )}
            </section>

            <section className="bg-blue-50/50 p-8 rounded-2xl border border-blue-100">
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1 pr-4">
                  <h2 className="text-xl font-semibold flex items-center gap-2 text-blue-900">
                    <BrainCircuit className="w-6 h-6 text-blue-600" />
                    AI Semantic Analysis & Transcription
                  </h2>
                  <p className="text-blue-700/80 mt-1">Use Gemini 3.1 Pro to detect speakers, generate a highly accurate transcript, and run content safety checks.</p>
                </div>
                <button 
                  onClick={runAiAnalysis}
                  disabled={isAiAnalyzing || isGeneratingJson}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap"
                >
                  {(isAiAnalyzing || isGeneratingJson) ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white"></div>
                      Analyzing...
                    </>
                  ) : 'Run AI Analysis'}
                </button>
              </div>

              {aiError && (
                <div className="p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3 border border-red-100 mb-6">
                  <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
                  <p>{aiError}</p>
                </div>
              )}

              {aiJsonResult && (
                <div className="flex flex-col sm:flex-row gap-4 mt-6">
                  <button
                    onClick={() => {
                      const blob = new Blob([aiJsonResult], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${file?.name.split('.')[0] || 'audio'}_analysis.json`;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }}
                    className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors shadow-sm w-full sm:w-auto"
                  >
                    <Download className="w-5 h-5" />
                    Download JSON Analysis
                  </button>
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string, value: string | number }) {
  return (
    <div className="bg-white p-5 rounded-xl border border-neutral-200 shadow-sm">
      <p className="text-sm font-medium text-neutral-500 mb-1">{label}</p>
      <p className="text-2xl font-semibold text-neutral-900">{value}</p>
    </div>
  );
}

function StatusCard({ label, isWarning, value }: { label: string, isWarning: boolean, value: string | number }) {
  return (
    <div className={`bg-white p-5 rounded-xl border shadow-sm ${isWarning ? 'border-red-200 bg-red-50' : 'border-neutral-200'}`}>
      <div className="flex items-start justify-between mb-2">
        <p className={`text-sm font-medium ${isWarning ? 'text-red-700' : 'text-neutral-500'}`}>{label}</p>
        {isWarning ? <AlertTriangle className="w-5 h-5 text-red-500" /> : <CheckCircle className="w-5 h-5 text-emerald-500" />}
      </div>
      <p className={`text-xl font-semibold ${isWarning ? 'text-red-900' : 'text-neutral-900'}`}>{value}</p>
    </div>
  );
}

function ScoreCard({ label, value, threshold, isWarning, isInteger = false }: { label: string, value: any, threshold: string, isWarning: boolean, isInteger?: boolean }) {
  const numValue = Number(value) || 0;
  return (
    <div className={`p-4 rounded-xl border shadow-sm ${isWarning ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
      <p className={`text-xs font-semibold uppercase tracking-wider mb-1 ${isWarning ? 'text-red-700' : 'text-emerald-700'}`}>
        {label}
      </p>
      <div className="flex items-center gap-2 mb-1">
        {isWarning ? <AlertTriangle className="w-5 h-5 text-red-600" /> : <CheckCircle className="w-5 h-5 text-emerald-600" />}
        <p className={`text-2xl font-bold ${isWarning ? 'text-red-900' : 'text-emerald-900'}`}>
          {isInteger ? Math.round(numValue) : numValue.toFixed(2)}
        </p>
      </div>
      <p className={`text-xs font-medium ${isWarning ? 'text-red-600' : 'text-emerald-600'}`}>
        {isWarning ? `Fail (${threshold})` : 'Pass'}
      </p>
    </div>
  );
}
