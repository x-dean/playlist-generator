import React, { useState } from 'react';
import { 
  Container, 
  Title, 
  Stack,
  Card,
  Text,
  Group,
  Button,
  Progress,
  Alert,
  Badge,
  List,
  Code,
  Grid
} from '@mantine/core';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  IconAnalyze, 
  IconMusic, 
  IconClock,
  IconChartLine,
  IconBrain,
  IconInfoCircle,
  IconPlayerPlay
} from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { apiClient } from '../api/client';

interface AnalysisJob {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  tracks_processed: number;
  total_tracks: number;
  created_at: string;
  completed_at?: string;
}

export const Analysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const queryClient = useQueryClient();

  // Fetch analysis status
  const { data: analysisJobs, isLoading } = useQuery<AnalysisJob[]>({
    queryKey: ['analysis-jobs'],
    queryFn: () => apiClient.get('/api/analysis/jobs').then(res => res.data),
    refetchInterval: isAnalyzing ? 2000 : false, // Poll while analyzing
  });

  // Start analysis mutation
  const startAnalysisMutation = useMutation({
    mutationFn: (options: { quick?: boolean; limit?: number }) => 
      apiClient.post('/api/analysis/start', options),
    onSuccess: () => {
      setIsAnalyzing(true);
      notifications.show({
        title: 'Analysis Started',
        message: 'Music analysis has been started successfully',
        color: 'green',
      });
      queryClient.invalidateQueries({ queryKey: ['analysis-jobs'] });
    },
    onError: (error: any) => {
      notifications.show({
        title: 'Analysis Failed',
        message: error.response?.data?.detail || 'Failed to start analysis',
        color: 'red',
      });
    },
  });

  const activeJob = analysisJobs?.find(job => job.status === 'running');
  const lastCompletedJob = analysisJobs?.find(job => job.status === 'completed');

  const handleStartAnalysis = (quick = false) => {
    startAnalysisMutation.mutate({ 
      quick, 
      limit: quick ? 5 : undefined 
    });
  };

  React.useEffect(() => {
    if (activeJob?.status !== 'running' && isAnalyzing) {
      setIsAnalyzing(false);
    }
  }, [activeJob, isAnalyzing]);

  return (
    <Container size="lg" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between" align="center">
          <div>
            <Title order={1} c="white">
              🔬 Audio Analysis
            </Title>
            <Text c="dimmed" size="lg">
              Analyze your music collection with ML-powered features
            </Text>
          </div>
        </Group>

        {/* Analysis Controls */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">Start Analysis</Title>
          <Text mb="lg" c="dimmed">
            Analyze your music files to extract comprehensive audio features including tempo, 
            key, mood, genre, and 25+ other characteristics.
          </Text>
          
          <Group gap="md">
            <Button 
              leftSection={<IconPlayerPlay size={16} />}
              onClick={() => handleStartAnalysis(true)}
              loading={startAnalysisMutation.isPending}
              disabled={isAnalyzing}
              variant="filled"
              color="blue"
            >
              Quick Test (5 files)
            </Button>
            <Button 
              leftSection={<IconAnalyze size={16} />}
              onClick={() => handleStartAnalysis(false)}
              loading={startAnalysisMutation.isPending}
              disabled={isAnalyzing}
              variant="filled"
              color="green"
            >
              Analyze All Files
            </Button>
          </Group>
        </Card>

        {/* Active Analysis */}
        {activeJob && (
          <Card withBorder padding="lg">
            <Group justify="space-between" mb="md">
              <Title order={3}>Analysis in Progress</Title>
              <Badge color="blue" variant="filled">Running</Badge>
            </Group>
            
            <Stack gap="md">
              <Text>
                Processing {activeJob.tracks_processed} of {activeJob.total_tracks} tracks
              </Text>
              <Progress 
                value={(activeJob.tracks_processed / activeJob.total_tracks) * 100} 
                size="lg" 
                animated
              />
              <Text size="sm" c="dimmed">
                Started: {new Date(activeJob.created_at).toLocaleString()}
              </Text>
            </Stack>
          </Card>
        )}

        {/* Analysis Results */}
        {lastCompletedJob && (
          <Card withBorder padding="lg">
            <Group justify="space-between" mb="md">
              <Title order={3}>Last Analysis Results</Title>
              <Badge color="green" variant="filled">Completed</Badge>
            </Group>
            
            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Group gap="sm" mb="xs">
                  <IconMusic size={20} color="#4DABF7" />
                  <Text fw={500}>Tracks Processed</Text>
                </Group>
                <Text size="xl" fw={700} c="blue">
                  {lastCompletedJob.tracks_processed}
                </Text>
              </Grid.Col>
              
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Group gap="sm" mb="xs">
                  <IconClock size={20} color="#51CF66" />
                  <Text fw={500}>Completion Time</Text>
                </Group>
                <Text size="sm">
                  {lastCompletedJob.completed_at ? 
                    new Date(lastCompletedJob.completed_at).toLocaleString() : 
                    'Unknown'
                  }
                </Text>
              </Grid.Col>
            </Grid>
          </Card>
        )}

        {/* Features Overview */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">Extracted Features</Title>
          <Text mb="lg" c="dimmed">
            Our analysis engine extracts comprehensive audio features using advanced ML models:
          </Text>
          
          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Group gap="xs" mb="sm">
                <IconChartLine size={16} color="#4DABF7" />
                <Text fw={500}>Basic Features</Text>
              </Group>
              <List size="sm" spacing="xs">
                <List.Item>Tempo (BPM)</List.Item>
                <List.Item>Key signature</List.Item>
                <List.Item>Loudness & dynamics</List.Item>
                <List.Item>Duration & structure</List.Item>
              </List>
            </Grid.Col>
            
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Group gap="xs" mb="sm">
                <IconBrain size={16} color="#51CF66" />
                <Text fw={500}>Advanced Features</Text>
              </Group>
              <List size="sm" spacing="xs">
                <List.Item>Spectral characteristics</List.Item>
                <List.Item>Harmonic analysis</List.Item>
                <List.Item>Rhythm patterns</List.Item>
                <List.Item>Timbral qualities</List.Item>
              </List>
            </Grid.Col>
          </Grid>
        </Card>

        {/* ML Models Info */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">ML Models</Title>
          <Text mb="lg" c="dimmed">
            Playlista v2 uses state-of-the-art machine learning models for audio analysis:
          </Text>
          
          <Stack gap="md">
            <Group gap="md">
              <Badge color="blue" variant="outline">Genre Classifier</Badge>
              <Text size="sm">50-class genre classification with 87% accuracy</Text>
            </Group>
            
            <Group gap="md">
              <Badge color="green" variant="outline">Mood Analyzer</Badge>
              <Text size="sm">3-dimensional mood analysis (valence, energy, danceability)</Text>
            </Group>
            
            <Group gap="md">
              <Badge color="orange" variant="outline">Audio Embeddings</Badge>
              <Text size="sm">512-dimensional feature vectors for similarity analysis</Text>
            </Group>
            
            <Group gap="md">
              <Badge color="purple" variant="outline">Feature Extractor</Badge>
              <Text size="sm">Comprehensive audio feature extraction using Librosa</Text>
            </Group>
          </Stack>
        </Card>

        {/* Performance Notes */}
        <Alert 
          icon={<IconInfoCircle size={16} />} 
          title="Performance Information" 
          color="blue"
        >
          <Text size="sm">
            Analysis typically takes 45-67 seconds per track depending on file size and duration. 
            For large libraries, consider starting with a quick test to verify functionality.
          </Text>
        </Alert>

        {/* Technical Details */}
        <Card withBorder padding="lg">
          <Title order={3} mb="md">Technical Details</Title>
          <Text mb="md" c="dimmed">
            Analysis pipeline specifications:
          </Text>
          
          <Stack gap="sm">
            <Group gap="md">
              <Code>Sample Rate:</Code>
              <Text size="sm">22,050 Hz (optimized for analysis)</Text>
            </Group>
            
            <Group gap="md">
              <Code>Frame Size:</Code>
              <Text size="sm">2048 samples</Text>
            </Group>
            
            <Group gap="md">
              <Code>Hop Length:</Code>
              <Text size="sm">512 samples</Text>
            </Group>
            
            <Group gap="md">
              <Code>Feature Count:</Code>
              <Text size="sm">27+ comprehensive audio features</Text>
            </Group>
          </Stack>
        </Card>
      </Stack>
    </Container>
  );
};