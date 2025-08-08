import React from 'react';
import { Container, Title, Button, Progress, Text, Group, Card } from '@mantine/core';
import { IconPlay, IconStop } from '@tabler/icons-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';

export function Analysis() {
  const { data: status, refetch } = useQuery({
    queryKey: ['analysis-status'],
    queryFn: () => api.get('/analysis/status').then(res => res.data),
    refetchInterval: 2000 // Refresh every 2 seconds
  });

  const handleStartAnalysis = async () => {
    try {
      await api.post('/analysis/start');
      refetch();
    } catch (error) {
      console.error('Failed to start analysis:', error);
    }
  };

  const handleStopAnalysis = async () => {
    try {
      await api.post('/analysis/stop');
      refetch();
    } catch (error) {
      console.error('Failed to stop analysis:', error);
    }
  };

  const isRunning = status?.engine_status === 'running';

  return (
    <Container size="xl">
      <Group justify="space-between" mb="xl">
        <Title order={1}>Audio Analysis</Title>
        <Group>
          {isRunning ? (
            <Button 
              leftSection={<IconStop size="1rem" />} 
              color="red"
              onClick={handleStopAnalysis}
            >
              Stop Analysis
            </Button>
          ) : (
            <Button 
              leftSection={<IconPlay size="1rem" />}
              onClick={handleStartAnalysis}
            >
              Start Analysis
            </Button>
          )}
        </Group>
      </Group>

      <Card shadow="sm" padding="lg" radius="md" withBorder mb="xl">
        <Text size="lg" fw={500} mb="md">Analysis Progress</Text>
        <Progress 
          value={status?.analysis_progress_percent || 0} 
          size="xl" 
          radius="md"
          mb="sm"
        />
        <Text size="sm" c="dimmed">
          {status?.analysis_progress_percent?.toFixed(1) || 0}% complete
        </Text>
      </Card>

      <Group grow mb="xl">
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text size="sm" c="dimmed">Total Tracks</Text>
          <Text size="xl" fw={700}>{status?.total_tracks || 0}</Text>
        </Card>

        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text size="sm" c="dimmed">Analyzed</Text>
          <Text size="xl" fw={700}>
            {status?.track_status_breakdown?.analyzed || 0}
          </Text>
        </Card>

        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text size="sm" c="dimmed">Queued</Text>
          <Text size="xl" fw={700}>
            {status?.job_status_breakdown?.queued || 0}
          </Text>
        </Card>

        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text size="sm" c="dimmed">Processing</Text>
          <Text size="xl" fw={700}>
            {status?.job_status_breakdown?.processing || 0}
          </Text>
        </Card>
      </Group>

      {status?.processing_statistics && (
        <Card shadow="sm" padding="lg" radius="md" withBorder>
          <Text size="lg" fw={500} mb="md">Performance Statistics</Text>
          <Text size="sm">
            Average processing time: {status.processing_statistics.average_processing_time_seconds}s
          </Text>
          <Text size="sm">
            Recent jobs analyzed: {status.processing_statistics.recent_jobs_count}
          </Text>
        </Card>
      )}
    </Container>
  );
}
