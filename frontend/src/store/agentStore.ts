import { create } from 'zustand';
import { AgentState, AgentAnalysisRequest, AgentAnalysisResponse } from '@/types';
import { apiService } from '@/services/apiService';
import { webSocketService } from '@/services/webSocketService';
import toast from 'react-hot-toast';

export const useAgentStore = create<AgentState>((set, get) => ({
  analyses: [],
  currentAnalysis: null,
  isLoading: false,
  error: null,

  requestAnalysis: async (request: AgentAnalysisRequest): Promise<AgentAnalysisResponse> => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.post<AgentAnalysisResponse>('/agents/analyze', request);
      const analysis = response.data;
      
      set(state => ({
        analyses: [analysis, ...state.analyses.slice(0, 49)], // Keep last 50 analyses
        currentAnalysis: analysis,
        isLoading: false
      }));
      
      const agentText = request.agents.length === 1 
        ? request.agents[0] 
        : `${request.agents.length} agents`;
      
      toast.success(
        `Analysis completed using ${agentText} for ${request.symbols.join(', ')} 
         (Cost: ${analysis.total_cost_usd.toFixed(2)})`
      );
      
      return analysis;
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Analysis request failed';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
      throw error;
    }
  },

  clearCurrentAnalysis: () => set({ currentAnalysis: null }),

  clearError: () => set({ error: null }),

  getAnalysisById: (requestId: string): AgentAnalysisResponse | null => {
    return get().analyses.find(a => a.request_id === requestId) || null;
  },

  updateAnalysis: (analysis: AgentAnalysisResponse) => {
    set(state => ({
      analyses: state.analyses.map(a => 
        a.request_id === analysis.request_id ? analysis : a
      ),
      currentAnalysis: state.currentAnalysis?.request_id === analysis.request_id 
        ? analysis 
        : state.currentAnalysis
    }));
  },
}));

// Initialize agent WebSocket listeners
webSocketService.on('analysis_result', (data: AgentAnalysisResponse) => {
  useAgentStore.getState().updateAnalysis(data);
  
  // Show notification if this is a new analysis
  const existingAnalysis = useAgentStore.getState().getAnalysisById(data.request_id);
  if (!existingAnalysis) {
    toast.success(`New analysis completed for ${data.symbols.join(', ')}`);
  }
});
