/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Typography, Box, Button, Paper,
  List, ListItem, ListItemText, ListItemIcon,
  Chip, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, Grid, Drawer
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { Analytics } from './Analytics';
import { NewEvaluation } from './NewEvaluation';

/**
 * Component for real-time log streaming over Server-Sent Events (SSE).
 *
 * Connects to `/api/runs/{runId}/stream-logs` and displays live log lines
 * and parsed scenario test pass/fail events.
 */
function LiveStream({ runId }: { runId: string }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [liveResults, setLiveResults] = useState<{scenario: string, passed: boolean}[]>([]);

  useEffect(() => {
    const es = new EventSource(`/api/runs/${runId}/stream-logs`);
    es.onmessage = (event) => {
      if (event.data === '[PROCESS_COMPLETED]') {
        es.close();
        return;
      }
      setLogs(prev => [...prev.slice(-49), event.data]);
      
      if (event.data.includes('✅ PASSED:')) {
        const scenario = event.data.split('✅ PASSED:')[1].trim();
        setLiveResults(prev => [...prev, { scenario, passed: true }]);
      } else if (event.data.includes('❌ FAILED:')) {
        const scenario = event.data.split('❌ FAILED:')[1].trim();
        setLiveResults(prev => [...prev, { scenario, passed: false }]);
      }
    };
    return () => es.close();
  }, [runId]);

  return (
    <Box>
      <Box sx={{ mb: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {liveResults.map((r, i) => (
          <Chip key={i} size="small" label={r.scenario} color={r.passed ? 'success' : 'error'} />
        ))}
      </Box>
      {logs.map((l, i) => <div key={i}>{l}</div>)}
    </Box>
  );
}

interface Turn {
  speaker: string;
  role: 'user' | 'assistant' | 'system';
  text: string;
}

function parseDetails(details: string): { parsed: Turn[] | null; raw: string } {
  if (!details) return { parsed: null, raw: '' };

  const blocks = details.split('\n\n').map(b => b.trim()).filter(b => b.length > 0);
  const parsedTurns: Turn[] = [];
  let hasValidParse = false;

  for (const block of blocks) {
    const match = block.match(/^(USER|ASSISTANT)\s*([^:]*?)\s*:\s*([\s\S]*)$/i);
    if (match) {
      const roleStr = match[1].toUpperCase();
      const speaker = match[2].trim() || (roleStr === 'USER' ? 'User' : 'Agent');
      const text = match[3].trim();
      parsedTurns.push({
        role: roleStr === 'USER' ? 'user' : 'assistant',
        speaker,
        text
      });
      hasValidParse = true;
    } else {
      parsedTurns.push({
        role: 'system',
        speaker: 'System',
        text: block
      });
    }
  }

  if (hasValidParse) {
    return { parsed: parsedTurns, raw: details };
  }

  return { parsed: null, raw: details };
}

interface LiveProgressState {
  runId: string;
  modelName: string;
  total: number;
  passed: number;
  failed: number;
  completedScenarios: { scenario: string; passed: boolean }[];
  logs: string[];
}

function parseMessageContent(text: string): { type: 'text' | 'tool_call' | 'tool_response'; content: string }[] {
  const regex = /(<\|tool_call>[\s\S]*?<tool_call\|>|<\|tool_response>[\s\S]*?<tool_response\|>)/g;
  const parts = text.split(regex);

  return parts.map(part => {
    if (part.startsWith('<|tool_call>')) {
      const content = part.replace('<|tool_call>', '').replace('<tool_call|>', '').trim();
      return { type: 'tool_call' as const, content };
    } else if (part.startsWith('<|tool_response>')) {
      const content = part.replace('<|tool_response>', '').replace('<tool_response|>', '').trim();
      return { type: 'tool_response' as const, content };
    }
    return { type: 'text' as const, content: part };
  }).filter(p => p.content.length > 0);
}

function renderParsedDetails(details: string) {
  const { parsed, raw } = parseDetails(details);

  if (!parsed) {
    return (
      <Box
        component="pre"
        sx={{
          m: 0, p: 2, bgcolor: 'grey.100', borderRadius: 1,
          fontFamily: 'monospace', fontSize: '0.85rem', overflowX: 'auto',
          whiteSpace: 'pre-wrap', wordBreak: 'break-all'
        }}
      >
        {raw}
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
      {parsed.map((turn, tIdx) => {
        if (turn.role === 'system') {
          return (
            <Typography
              key={tIdx}
              variant="caption"
              align="center"
              sx={{ color: 'text.secondary', fontStyle: 'italic', display: 'block', my: 0.5 }}
            >
              {turn.text}
            </Typography>
          );
        }

        const isUser = turn.role === 'user';
        return (
          <Box
            key={tIdx}
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignSelf: isUser ? 'flex-start' : 'flex-end',
              maxWidth: '80%',
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 0.5, px: 1, alignSelf: isUser ? 'flex-start' : 'flex-end' }}>
              {turn.speaker}
            </Typography>
            <Box
              sx={{
                p: 1.5,
                borderRadius: 2,
                bgcolor: isUser ? 'primary.main' : 'success.main',
                color: '#fff',
                boxShadow: 1,
                borderTopLeftRadius: isUser ? 0 : 2,
                borderTopRightRadius: isUser ? 2 : 0,
              }}
            >
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {parseMessageContent(turn.text).map((c, cIdx) => {
                  if (c.type === 'tool_call') {
                    return (
                      <Box
                        key={cIdx}
                        sx={{
                          p: 1,
                          bgcolor: 'rgba(255, 255, 255, 0.15)',
                          border: '1px dashed rgba(255, 255, 255, 0.5)',
                          borderRadius: 1,
                          fontFamily: 'monospace',
                          fontSize: '0.8rem',
                          color: '#fff',
                          wordBreak: 'break-all',
                          my: 0.5
                        }}
                      >
                        🛠️ <strong>Tool Call:</strong> {c.content}
                      </Box>
                    );
                  }
                  if (c.type === 'tool_response') {
                    return (
                      <Box
                        key={cIdx}
                        sx={{
                          p: 1,
                          bgcolor: 'rgba(0, 0, 0, 0.05)',
                          border: '1px dashed rgba(0, 0, 0, 0.2)',
                          borderRadius: 1,
                          fontFamily: 'monospace',
                          fontSize: '0.8rem',
                          color: 'text.secondary',
                          wordBreak: 'break-all',
                          my: 0.5
                        }}
                      >
                        📥 <strong>Tool Response:</strong> {c.content}
                      </Box>
                    );
                  }
                  return (
                    <Typography key={cIdx} variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {c.content}
                    </Typography>
                  );
                })}
              </Box>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}

/**
 * Primary dashboard view for GBench UI.
 *
 * Displays overview cards, historical benchmark runs, active live-stream logs,
 * and quick-launch modal triggers for new evaluation jobs.
 */
export function Dashboard() {
  const [runs, setRuns] = useState<any[]>([]);
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [activeScenario, setActiveScenario] = useState<any | null>(null);
  const [isNewEvalOpen, setIsNewEvalOpen] = useState(false);

  useEffect(() => {
    fetchRuns();
    // Poll runs every 5 seconds
    const interval = setInterval(fetchRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchRuns = async () => {
    try {
      const res = await axios.get('/api/runs');
      setRuns(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const runningRunId = runs.find(r => r.status === 'running')?.run_id;

  const [liveProgress, setLiveProgress] = useState<LiveProgressState | null>(null);

  useEffect(() => {
    if (!runningRunId) {
      setLiveProgress(null);
      return;
    }

    const runningRun = runs.find(r => r.run_id === runningRunId);
    const totalScenarios = runningRun?.total_scenarios || runningRun?.results?.quality?.raw_results?.counts?.total || 0;

    setLiveProgress({
      runId: runningRunId,
      modelName: runningRun?.model_name || 'gemma-4-E2B-it',
      total: totalScenarios,
      passed: 0,
      failed: 0,
      completedScenarios: [],
      logs: []
    });

    const es = new EventSource(`/api/runs/${runningRunId}/stream-logs`);
    
    es.onerror = (err) => {
      console.error("React EventSource error:", err);
    };
    
    es.onmessage = (event) => {
      console.log("React EventSource received:", event.data);
      if (event.data === '[PROCESS_COMPLETED]') {
        es.close();
        fetchRuns();
        return;
      }

      setLiveProgress(prev => {
        if (!prev || prev.runId !== runningRunId) return prev;
        
        const newLogs = [...(prev.logs || []), event.data].slice(-10);
        
        let passedInc = 0;
        let failedInc = 0;
        let newCompleted = [...prev.completedScenarios];
        
        if (event.data.includes('✅ PASSED:')) {
          const scenario = event.data.split('✅ PASSED:')[1].trim();
          if (!newCompleted.some(s => s.scenario === scenario)) {
            passedInc = 1;
            newCompleted.push({ scenario, passed: true });
          }
        } else if (event.data.includes('❌ FAILED:')) {
          const scenario = event.data.split('❌ FAILED:')[1].trim();
          if (!newCompleted.some(s => s.scenario === scenario)) {
            failedInc = 1;
            newCompleted.push({ scenario, passed: false });
          }
        }
        
        return {
          ...prev,
          logs: newLogs,
          passed: prev.passed + passedInc,
          failed: prev.failed + failedInc,
          completedScenarios: newCompleted
        };
      });
    };

    return () => {
      console.log("Closing EventSource for run ID:", runningRunId);
      es.close();
    };
  }, [runningRunId]);

  return (
    <Box sx={{ p: 0 }}>
      <Grid container spacing={4} sx={{ alignItems: 'stretch' }}>
        {/* Left Side: Analytics */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Analytics liveProgress={liveProgress} />
        </Grid>

        {/* Right Side: Evaluation Runs */}
        <Grid size={{ xs: 12, lg: 6 }}>
          <Paper sx={{ p: 4, height: '100%', boxSizing: 'border-box', border: '1px solid #e0e0e0', boxShadow: 'none', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, flexWrap: 'wrap', gap: 1.5 }}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>Evaluation Runs</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button size="small" variant="outlined" onClick={fetchRuns}>Refresh</Button>
                <Button size="small" variant="contained" onClick={() => setIsNewEvalOpen(true)}>New</Button>
              </Box>
            </Box>

            {runs.length === 0 ? (
              <Box sx={{ p: 6, textAlign: 'center', backgroundColor: '#f8f9fa', borderRadius: 2, border: '1px dashed #ccc', flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography color="textSecondary">No benchmarks recorded. Click "New" to begin.</Typography>
              </Box>
            ) : (
              <Box sx={{ overflowY: 'auto', flex: 1, maxHeight: 950 }}>
                <List disablePadding>
                  {runs.map(run => (
                    <ListItem 
                      key={run.run_id} 
                      divider 
                      sx={{ flexDirection: 'column', alignItems: 'stretch', py: 0, px: 0 }}
                    >
                      {/* Clickable Header Area */}
                      <Box 
                        onClick={() => setExpandedRun(expandedRun === run.run_id ? null : run.run_id)}
                        sx={{ 
                          cursor: 'pointer',
                          py: 2.5, 
                          px: 2, 
                          display: 'flex', 
                          flexDirection: 'column',
                          '&:hover': { backgroundColor: '#fdfdfd' }
                        }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', mb: 1 }}>
                          <Box>
                            <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'primary.main', fontSize: '1rem' }}>
                              {run.model_name || `Benchmark Run`}
                            </Typography>
                            <Typography variant="caption" color="textSecondary" sx={{ fontFamily: 'monospace' }}>
                              {run.run_id}
                            </Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            {run.results?.quality?.raw_results?.counts && (
                              <Box sx={{ textAlign: 'right' }}>
                                <Typography variant="body2" sx={{ fontWeight: 700, color: run.results.quality.raw_results.counts.passed === run.results.quality.raw_results.counts.total ? 'success.main' : 'warning.main', fontSize: '0.9rem' }}>
                                  {run.results.quality.raw_results.counts.total > 0 ? Math.round(run.results.quality.raw_results.counts.passed / run.results.quality.raw_results.counts.total * 100) : 0}%
                                </Typography>
                                <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.75rem' }}>
                                  {run.results.quality.raw_results.counts.passed}/{run.results.quality.raw_results.counts.total}
                                </Typography>
                              </Box>
                            )}
                            <Chip 
                              label={run.status.toUpperCase()} 
                              size="small"
                              color={run.status === 'completed' ? 'success' : run.status === 'running' ? 'primary' : 'default'} 
                              sx={{ fontWeight: 600, fontSize: '0.7rem', height: 20 }}
                            />
                          </Box>
                        </Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="caption" color="textSecondary">
                            {new Date(run.created_at).toLocaleString()}
                          </Typography>
                          {run.tags && run.tags.length > 0 && (
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {run.tags.map((t: string) => <Chip key={t} label={t} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.65rem' }} />)}
                            </Box>
                          )}
                        </Box>
                      </Box>
                      
                      {/* Expandable Content Area */}
                      <Box sx={{ display: expandedRun === run.run_id ? 'block' : 'none', width: '100%', px: 2, pb: 2, bgcolor: '#fafafa', borderTop: '1px solid #f0f0f0' }}>
                        {run.status === 'running' ? (
                          <Box sx={{ width: '100%' }}>
                            <Box sx={{ mt: 2, width: '100%', bgcolor: '#1e1e1e', color: '#a6e22e', p: 2, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.85rem', maxHeight: 200, overflow: 'auto' }}>
                              <LiveStream runId={run.run_id} />
                            </Box>
                            <Button 
                              size="small" 
                              variant="outlined" 
                              color="error" 
                              sx={{ mt: 1.5 }}
                              onClick={async () => {
                                try {
                                  await axios.post(`/api/runs/${run.run_id}/cancel`);
                                  fetchRuns();
                                } catch (e) {
                                  console.error(e);
                                }
                              }}
                            >
                              Cancel Run
                            </Button>
                          </Box>
                        ) : run.results?.quality?.raw_results?.scenarios ? (
                          <Box sx={{ mt: 1.5 }}>
                            <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 600, fontSize: '0.85rem' }}>Scenario Details</Typography>
                            <Box sx={{ maxHeight: 250, overflow: 'auto', border: '1px solid #e0e0e0', borderRadius: 1, bgcolor: '#fff' }}>
                              <List dense disablePadding>
                                 {run.results.quality.raw_results.scenarios.map((s: any, idx: number) => (
                                  <ListItem
                                    key={idx}
                                    divider
                                    onClick={() => setActiveScenario(s)}
                                    sx={{
                                      '&:last-child': { borderBottom: 0 },
                                      cursor: 'pointer',
                                      '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.04)' }
                                    }}
                                  >
                                    <ListItemIcon sx={{ minWidth: 24, fontSize: '0.9rem' }}>
                                      {s.status === 'pass' ? '✅' : '❌'}
                                    </ListItemIcon>
                                    <ListItemText
                                      primary={
                                        <Typography sx={{ fontWeight: 500, fontSize: '0.8rem', color: s.status === 'pass' ? 'text.primary' : 'error.main' }}>
                                          {s.name}
                                        </Typography>
                                      }
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          </Box>
                        ) : (
                          <Typography variant="body2" color="textSecondary" sx={{ mt: 1.5, fontSize: '0.8rem' }}>
                            No detailed scenario results available for this run.
                          </Typography>
                        )}
                      </Box>
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Scenario Details Dialog */}
      <Dialog
        open={activeScenario !== null}
        onClose={() => setActiveScenario(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center', m: 0, p: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Scenario Details: {activeScenario?.name}</Typography>
          <IconButton onClick={() => setActiveScenario(null)}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ p: 3 }}>
          {activeScenario && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Overall Status:</Typography>
                <Chip
                  label={activeScenario.status.toUpperCase()}
                  color={activeScenario.status === 'pass' ? 'success' : 'error'}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
              </Box>

              {activeScenario.details && (
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Details / Transcript:</Typography>
                  {renderParsedDetails(activeScenario.details)}
                </Box>
              )}

              {activeScenario.steps && activeScenario.steps.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>Steps:</Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {activeScenario.steps.map((step: any, sIdx: number) => (
                      <Box key={sIdx} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2, bgcolor: '#fafafa' }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {step.name}
                          </Typography>
                          <Chip
                            label={step.status.toUpperCase()}
                            color={step.status === 'pass' ? 'success' : step.status === 'fail' ? 'error' : 'default'}
                            size="small"
                          />
                        </Box>
                        {step.details && (
                          <Box sx={{ mt: 1 }}>
                            {renderParsedDetails(step.details)}
                          </Box>
                        )}
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActiveScenario(null)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Drawer
        anchor="right"
        open={isNewEvalOpen}
        onClose={() => setIsNewEvalOpen(false)}
        sx={{ zIndex: 1300 }}
        slotProps={{
          paper: {
            sx: {
              width: '66.6vw',
            }
          }
        }}
      >
        <Box sx={{ p: 4, height: '100%', boxSizing: 'border-box', overflowY: 'auto' }}>
          <NewEvaluation
            onClose={() => setIsNewEvalOpen(false)}
            onRunTriggered={() => {
              setIsNewEvalOpen(false);
              fetchRuns();
            }}
          />
        </Box>
      </Drawer>
    </Box>
  );
}
