import { notFound } from "next/navigation";
import PlayerReviewView from "../../../components/player-review-view";
import ProjectionReviewView from "../../../components/projection-review-view";
import TeamReviewView from "../../../components/team-review-view";
import TournamentReviewView from "../../../components/tournament-review-view";

type PlayerTeamReviewPageProps = {
  params: Promise<{ view: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

export default async function PlayerTeamReviewPage({ params, searchParams }: PlayerTeamReviewPageProps) {
  const resolvedParams = await params;
  const query = await searchParams;
  const view = String(resolvedParams.view || "").trim().toLowerCase();
  const selectedDate = typeof query.date === "string" ? query.date : getTodayIsoDate();

  if (view === "player-review") {
    return <PlayerReviewView selectedDate={selectedDate} />;
  }

  if (view === "team-review") {
    return <TeamReviewView selectedDate={selectedDate} />;
  }

  if (view === "projection-review") {
    return <ProjectionReviewView selectedDate={selectedDate} />;
  }

  if (view === "tournament-review") {
    const contestId = typeof query.contest_id === "string" ? query.contest_id : "contest";
    return <TournamentReviewView selectedDate={selectedDate} defaultContestId={contestId} />;
  }

  notFound();
}
