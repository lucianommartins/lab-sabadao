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

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { Compare } from './pages/Compare';
import { Analytics } from './pages/Analytics';
import { NewEvaluation } from './pages/NewEvaluation';

const theme = createTheme({
  palette: {
    background: {
      default: '#f8f9fa',
    },
    primary: {
      main: '#1a73e8', // Google blue
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
          borderRadius: 8,
          border: '1px solid #e0e0e0',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 6,
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        size: 'small',
        variant: 'outlined',
      },
    },
  },
});

/**
 * Root Application component for GBench UI dashboard.
 *
 * Configures global Material UI theme and client-side routing across
 * dashboard, comparison, analytics, and evaluation views.
 */
export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="compare" element={<Compare />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="new-evaluation" element={<NewEvaluation />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
